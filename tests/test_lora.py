import torch
import torch.nn as nn

from finetune_esm.models import lora


def test_scaling_is_alpha_over_rank_not_alpha():
    layer = lora.LoRALayer(4, 3, rank=8, alpha=16)
    assert layer.scaling == 2.0


def test_output_starts_at_zero_because_w_b_is_zero_initialised():
    layer = lora.LoRALayer(4, 3, rank=8, alpha=16)
    assert torch.allclose(layer(torch.randn(5, 4)), torch.zeros(5, 3))


def test_forward_equals_base_plus_scaled_low_rank_update():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    wrapped = lora.LinearWithLoRA(base, rank=2, alpha=4)
    with torch.no_grad():
        wrapped.lora.W_b.copy_(torch.randn(2, 3))

    x = torch.randn(6, 4)
    expected = base(x) + 2.0 * (x @ wrapped.lora.W_a @ wrapped.lora.W_b)
    assert torch.allclose(wrapped(x), expected, atol=1e-6)


def test_freeze_base_controls_base_weight_gradients():
    frozen = lora.LinearWithLoRA(nn.Linear(4, 3), rank=2, alpha=4, freeze_base=True)
    assert not frozen.linear.weight.requires_grad
    assert frozen.lora.W_a.requires_grad

    trainable = lora.LinearWithLoRA(nn.Linear(4, 3), rank=2, alpha=4, freeze_base=False)
    assert trainable.linear.weight.requires_grad


def test_head_base_weights_remain_trainable_after_apply_lora():
    """Regression: the old implementation froze a randomly-initialised head."""
    model = nn.Module()
    model.head = nn.Linear(4, 3)
    model.body = nn.Linear(4, 4)
    lora.apply_lora(model, rank=2, alpha=4, target_layers=["body"], head_layers=["head"])
    assert model.head.linear.weight.requires_grad
    assert not model.body.linear.weight.requires_grad


def test_matches_peft_numerically_under_identical_initialisation():
    # Adapted for the installed peft==0.20.0, whose `lora.Linear.__init__`
    # signature is (base_layer, adapter_name, config, r, lora_alpha,
    # is_target_conv_1d_layer, **kwargs) — it requires a `LoraConfig`
    # instance that the brief's original construction did not pass.
    from peft.tuners.lora import Linear as PeftLoraLinear
    from peft.tuners.lora import LoraConfig

    torch.manual_seed(0)
    base = nn.Linear(8, 5)
    ours = lora.LinearWithLoRA(base, rank=4, alpha=8)
    with torch.no_grad():
        ours.lora.W_b.copy_(torch.randn(4, 5))

    peft_config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0)
    peft_layer = PeftLoraLinear(
        base_layer=nn.Linear(8, 5),
        adapter_name="default",
        config=peft_config,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    with torch.no_grad():
        peft_layer.base_layer.weight.copy_(base.weight)
        peft_layer.base_layer.bias.copy_(base.bias)
        # ours: (in, r) and (r, out); peft: (r, in) and (out, r)
        peft_layer.lora_A["default"].weight.copy_(ours.lora.W_a.T)
        peft_layer.lora_B["default"].weight.copy_(ours.lora.W_b.T)

    x = torch.randn(7, 8)
    ours.eval()
    peft_layer.eval()
    with torch.no_grad():
        assert torch.allclose(ours(x), peft_layer(x), atol=1e-5)


def test_count_trainable_parameters():
    model = nn.Linear(4, 3)
    assert lora.count_trainable_parameters(model) == 4 * 3 + 3
    model.weight.requires_grad = False
    assert lora.count_trainable_parameters(model) == 3

import torch

from finetune_esm.models import heads


def test_output_is_one_scalar_per_row():
    head = heads.MLPHead(16)
    assert head(torch.randn(5, 16)).shape == (5,)


def test_hidden_dim_defaults_to_input_dim():
    head = heads.MLPHead(16)
    assert head.net[0].out_features == 16


def test_dropout_is_disabled_in_eval_mode():
    torch.manual_seed(0)
    head = heads.MLPHead(16, dropout=0.9).eval()
    x = torch.randn(4, 16)
    assert torch.allclose(head(x), head(x))


def test_gradients_flow_to_all_parameters():
    head = heads.MLPHead(16)
    head(torch.randn(3, 16)).sum().backward()
    for name, param in head.named_parameters():
        assert param.grad is not None, name

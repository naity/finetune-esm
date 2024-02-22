import math
import torch
import torch.nn as nn
from functools import partial


class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = x @ self.W_a  # batch * rank
        x = x @ self.W_b  # batch * out_dim
        return self.alpha * x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def apply_lora(
    model,
    lora_rank=8,
    lora_alpha=16,
    lora_query=True,
    lora_key=False,
    lora_value=True,
    lora_projection=False,
    lora_mlp=False,
    lora_head=True,
):
    # freeze model layers
    for param in model.parameters():
        param.requires_grad = False

    # for adding lora to linear layers
    linear_with_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)

    # iterate through each transfomer layer
    for layer in model.llm.encoder.layer:
        if lora_query:
            layer.attention.self.query = linear_with_lora(layer.attention.self.query)

        if lora_key:
            layer.attention.self.key = linear_with_lora(layer.attention.self.key)

        if lora_value:
            layer.attention.self.value = linear_with_lora(layer.attention.self.value)

        if lora_projection:
            layer.attention.output.dense = linear_with_lora(
                layer.attention.output.dense
            )

        if lora_mlp:
            layer.output.dense = linear_with_lora(layer.output.dense)
            layer.output.dense = linear_with_lora(layer.output.dense)

    if lora_head:
        model.pre_classifier = linear_with_lora(model.pre_classifier)
        model.classifier = linear_with_lora(model.classifier)

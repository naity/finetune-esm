import math
import torch
import torch.nn as nn
from functools import partial


class LoRALayer(nn.Module):
    """
    A LoRA (Low-Rank Adaptation) layer for efficient modeling.

    Implements a low-rank factorization of a linear layer for reduced
    computational cost and memory usage.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        rank (int): Rank of the low-rank factorization.
        alpha (float): Scalar factor for scaling the output.
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """

        x = x @ self.W_a  # batch * rank
        x = x @ self.W_b  # batch * out_dim
        return self.alpha * x


class LinearWithLoRA(nn.Module):
    """
    Combines a linear layer with a LoRAlayer for efficient modeling.

    Applies the linear laye and the LoRA layer to the input.

    Args:
        linear (nn.Module): An existing linear layer to be used.
        rank (int): Rank of the low-rank factorization in the LoRA layer.
        alpha (float): Scalar factor for scaling the output of the LoRA layer.
    """

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass through the combined linear and LoRA layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """

        return self.linear(x) + self.lora(x)


def apply_lora(
    model: nn.Module,
    lora_rank: int = 8,
    lora_alpha: float = 16,
    lora_query: bool = True,
    lora_key: bool = False,
    lora_value: bool = True,
    lora_projection: bool = False,
    lora_mlp: bool = False,
    lora_head: bool = True,
) -> None:
    """
    Applies LoRA layers to a given model's transformer layers and head layers.

    Modifies the model in-place by replacing specified linear layers with
    LinearWithLoRA layers for efficient modeling.

    Args:
        model (nn.Module): The model to apply LoRA to.
        lora_rank (int, optional): Rank of the low-rank factorization in LoRA layers. Defaults to 8.
        lora_alpha (float, optional): Scalar factor for scaling the output of LoRA layers. Defaults to 16.
        lora_query (bool, optional): Whether to apply LoRA to query projections. Defaults to True.
        lora_key (bool, optional): Whether to apply LoRA to key projections. Defaults to False.
        lora_value (bool, optional): Whether to apply LoRA to value projections. Defaults to True.
        lora_projection (bool, optional): Whether to apply LoRA to the attention output projection. Defaults to False.
        lora_mlp (bool, optional): Whether to apply LoRA to the MLP layers. Defaults to False.
        lora_head (bool, optional): Whether to apply LoRA to the head layers. Defaults to True.
    """

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
            layer.intermediate.dense = linear_with_lora(layer.intermediate.dense)
            layer.output.dense = linear_with_lora(layer.output.dense)

    if lora_head:
        model.pre_classifier = linear_with_lora(model.pre_classifier)
        model.classifier = linear_with_lora(model.classifier)


def freeze_all_but_head(
    model: nn.Module,
) -> None:
    """
    Freezes all layers of a model except for the head layers, allowing training
    to focus only on the head layers for fine-tuning or adaptation.

    Modifies the model in-place by setting `requires_grad=False` for all
    parameters except those in the `pre_classifier` and `classifier` layers.

    Args:
        model (nn.Module): The model to apply head-only training to.
    """

    # freeze model layers
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze head layers
    for param in model.pre_classifier.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

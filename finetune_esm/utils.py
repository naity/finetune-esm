import torch
import torch.nn as nn
import numpy as np

from typing import Dict, Any
from config import mlflow
from ray.train.torch import get_device
from transformers import AutoTokenizer, EsmModel
from ray.train.lightning import RayFSDPStrategy
from ray.train.lightning._lightning_utils import (
    _LIGHTNING_GREATER_EQUAL_2_0,
    _TORCH_FSDP_AVAILABLE,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)


def collate_fn(
    batch: dict[str, np.ndarray],
    tokenizer: AutoTokenizer,
    targets_dtype: torch.dtype = torch.float,
) -> dict[str, torch.Tensor]:
    """
    Collates a batch of sequences and targets into a format suitable for training.

    Args:
        batch (dict[str, np.ndarray]):
            A dictionary containing sequences and targets.
            - "input_ids": Numpy array of integer token IDs.
            - "attention_mask": Numpy array of attention mask values.
            - "targets": Numpy array of target values.
        tokenizer (transformers.AutoTokenizer): The tokenizer to use for padding.
        targets_dtype (torch.dtype) = Data type for the targets tensor.Defaults to torch.float.

    Returns:
        Dict[str, torch.Tensor]:
            A dictionary containing the padded and batched tensors:
            - "input_ids": Padded and batched tensor of token IDs.
            - "attention_mask": Padded and batched tensor of attention masks.
            - "targets": Tensor of targets, converted to `target_dtype` and moved to device.
    """

    padded = tokenizer.pad(
        {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        return_tensors="pt",
    )

    batch["input_ids"] = padded["input_ids"].to(device=get_device())
    batch["attention_mask"] = padded["attention_mask"].to(device=get_device())
    batch["targets"] = torch.as_tensor(
        batch["targets"], dtype=targets_dtype, device=get_device()
    )

    return batch


def get_loss_func(loss_func_name: str) -> nn.Module:
    """Retrieves a loss function based on the provided name.

    Args:
        loss_func_name (str): The name of the desired loss function. Supported options are:
            - "cross_entropy": Cross-entropy loss.
            - "mse": Mean squared error loss.

    Returns:
        nn.Module: An instance of the requested loss function, or nn.BCEWithLogitsLoss() if the name is not found.
    """

    if loss_func_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_func_name == "mse":
        return nn.MSELoss()
    else:
        return nn.BCEWithLogitsLoss()


def count_trainable_parameters(model: EsmModel) -> int:
    """
    Counts the number of trainable parameters in an ESM model.

    Args:
        model (EsmModel): The ESM model to count parameters for.

    Returns:
        int: The total number of trainable parameters in the model.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_run_id(experiment_name: str, trial_id: str) -> str:
    """
    Retrieves the MLflow run ID for a given experiment name and trial ID.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        trial_id (str): The ID of the trial within the experiment.

    Returns:
        str: The run ID of the matching MLflow run.
    """

    trial_name = f"TorchTrainer_{trial_id}"
    run = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.trial_name = '{trial_name}'",
    ).iloc[0]
    return run["run_id"]


class CustomRayFSDPStrategy(RayFSDPStrategy):
    """A custom RayFSDPStrategy to avoid unwanted truncation of `state_dict` keys"""

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """Gathers the full state dict to rank 0 on CPU."""
        assert self.model is not None, "Failed to get the state dict for a None model!"

        if _LIGHTNING_GREATER_EQUAL_2_0 and _TORCH_FSDP_AVAILABLE:
            with FullyShardedDataParallel.state_dict_type(
                module=self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                ),
            ):
                state_dict = self.model.state_dict()
                # replace "_forward_module." if present instead of using string splicing
                return {
                    k.replace("_forward_module.", ""): v for k, v in state_dict.items()
                }
        else:
            # Otherwise Lightning uses Fairscale FSDP, no need to unshard by ourself.
            return super().lightning_module_state_dict()

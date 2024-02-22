import ray
import time
import typer
import torch
import torch.nn as nn
import json
import functools
import datetime
import numpy as np
import lightning.pytorch as pl

from typing import Optional
from typing_extensions import Annotated
from data import load_data, CustomPreprocessor
from utils import collate_fn, get_run_id, get_loss_func
from models import FinetuneESM, ESMLightningModule
from transformers import AutoTokenizer
from config import STORAGE_DIR, MLFLOW_TRACKING_URI, logger
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from ray.train.lightning import RayFSDPStrategy
from transformers.models.esm.modeling_esm import EsmLayer
from ray.train import CheckpointConfig, DataConfig, RunConfig, ScalingConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

# create a typer app
app = typer.Typer()


def train_loop_per_worker(config: dict) -> None:
    """
    Trains a FinetuneESM model on a distributed Ray dataset using PyTorch Lightning.

    Args:
        config (dict): A configuration dictionary containing model hyperparameters,
            training settings, and Ray-specific configuration.

    Returns:
        None
    """

    esm_model = config["esm_model"]
    dropout_p = config["dropout_p"]
    num_classes = config["num_classes"]
    learning_rate = config["learning_rate"]
    num_epochs = config["num_epochs"]
    num_devices = config["num_devices"]
    batch_size_per_worker = config["batch_size_per_worker"]
    loss_func_name = config["loss_func_name"]
    score_name = config["score_name"]
    verbose = config["verbose"]

    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")

    tokenizer = AutoTokenizer.from_pretrained(esm_model)
    collate_fn_partial = functools.partial(collate_fn, tokenizer=tokenizer)

    train_loader = train_ds.iter_torch_batches(
        batch_size=batch_size_per_worker, collate_fn=collate_fn_partial
    )
    val_loader = val_ds.iter_torch_batches(
        batch_size=batch_size_per_worker, collate_fn=collate_fn_partial
    )

    model = FinetuneESM(
        esm_model=esm_model,
        dropout_p=dropout_p,
        num_classes=num_classes,
    )

    loss_fn = get_loss_func(loss_func_name)
    lightning_model = ESMLightningModule(
        model=model, learning_rate=learning_rate, loss_fn=loss_fn, score_name=score_name
    )

    callbacks = [RayTrainReportCallback()]

    if score_name == "multilabel_f1_score":
        # save the model with highest f1_score
        callbacks.append(
            ModelCheckpoint(save_top_k=1, mode="max", monitor="val_f1_score")
        )

    # use FSDP for training
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={EsmLayer}
    )
    strategy = RayFSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        auto_wrap_policy=auto_wrap_policy,
        limit_all_gathers=True,
        activation_checkpointing_policy={EsmLayer},
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        devices=num_devices,
        accelerator="cuda",
        precision="16-mixed",
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        callbacks=callbacks,
    )
    trainer = prepare_trainer(trainer)

    start = time.time()
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    end = time.time()

    if verbose:
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
        print(f"Training Time: {(end-start)/60:.2f} min")


@app.command()
def train_model(
    experiment_name: Annotated[
        str, typer.Option(help="Name for the training experiment")
    ],
    dataset_loc: Annotated[
        str, typer.Option(help="Path to the dataset in parquet format")
    ],
    targets_loc: Annotated[
        str, typer.Option(help="Path to the NumPy .npy file containing targets")
    ],
    num_samples: Annotated[
        Optional[int],
        typer.Option(help=" Number of samples to load from the dataset"),
    ] = None,
    val_size: Annotated[
        float, typer.Option(help="Proportion of the dataset to use for validation")
    ] = 0.25,
    esm_model: Annotated[
        str, typer.Option(help="ESM model name to use")
    ] = "esm2_t6_8M_UR50D",
    dropout_p: Annotated[
        float, typer.Option(help="Dropout probability for regularization")
    ] = 0.05,
    num_classes: Annotated[
        int, typer.Option(help="Number of final output dimensions")
    ] = 100,
    learning_rate: Annotated[
        float, typer.Option(help="The learning rate for the optimizer")
    ] = 1e-3,
    num_epochs: Annotated[int, typer.Option(help="Number of epochs for training")] = 3,
    num_devices: Annotated[
        int, typer.Option(help="Number of GPUs to use per worker")
    ] = 1,
    batch_size_per_worker: Annotated[
        int, typer.Option(help="Number of samples per batch for each worker")
    ] = 8,
    loss_func_name: Annotated[
        str, typer.Option(help="Training loss function name")
    ] = "bcewithlogits",
    score_name: Annotated[
        str, typer.Option(help="Score metric to track during training")
    ] = "multilabel_f1_score",
    num_workers: Annotated[
        int, typer.Option(help="Number of workers to use for training")
    ] = 1,
    verbose: Annotated[
        bool, typer.Option(help="Whether to print verbose training messages")
    ] = True,
) -> ray.train.Result:
    """Trains an ML model using the specified parameters.

    Args:
        experiment_name (str): Name for the training experiment.
        dataset_loc (str): Path to the dataset in parquet format.
        targets_loc (str): Path to the NumPy .npy file containing the targets for the samples in the dataset.
        num_samples (Optional[int], optional): Number of samples to load from the dataset. Defaults to None (loads all samples).
        val_size (float, optional): Proportion of the dataset to use for validation. Defaults to 0.25.
        esm_model (str, optional): ESM model name to use. Defaults to "esm2_t6_8M_UR50D".
        dropout_p (float, optional): Dropout probability for regularization. Defaults to 0.05.
        num_classes (int, optional): Number of final output dimensions. Defaults to 100.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        num_epochs (int, optional): Number of epochs for training. Defaults to 3.
        num_devices (int, optional): Number of GPUs to use per worker. Defaults to 1.
        batch_size_per_worker (int, optional): Number of samples per batch for each worker. Defaults to 3.
        loss_func_name (str, optional): Training loss function name. Defaults to "bcewithlogits".
        score_name (str, optional): Score metric to track during training. Defaults to "multilabel_f1_score".
        num_workers (int, optional): Number of workers to use for training. Defaults to 1.
        verbose (bool, optional): Whether to print verbose logging messages. Defaults to False.

    Returns:
        Dict: A ray.train.Result object containing training results.
    """
    # configs
    train_loop_config = {
        "dropout_p": dropout_p,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "num_devices": num_devices,
        "batch_size_per_worker": batch_size_per_worker,
        "loss_func_name": loss_func_name,
        "score_name": score_name,
        "verbose": verbose,
    }

    if not esm_model.startswith("facebook"):
        esm_model = "facebook/" + esm_model

    train_loop_config["esm_model"] = esm_model

    scaling_config = ScalingConfig(
        num_workers=num_workers,
        use_gpu=True,
    )

    checkpoint_config = CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    )

    # MLflow callback
    mlflow_callback = MLflowLoggerCallback(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=experiment_name,
        save_artifact=True,
    )

    run_config = RunConfig(
        callbacks=[mlflow_callback],
        storage_path=STORAGE_DIR,
        checkpoint_config=checkpoint_config,
        local_dir=STORAGE_DIR,
    )

    options = ray.data.ExecutionOptions(preserve_order=True)
    dataset_config = DataConfig(datasets_to_split=["train"], execution_options=options)

    # Dataset
    ds = load_data(dataset_loc, num_samples)
    train_ds, val_ds = ds.train_test_split(val_size)
    targets = np.load(targets_loc)
    preprocessor = CustomPreprocessor(esm_model, targets)
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    # Trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_loop_config,
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
        dataset_config=dataset_config,
    )

    results = trainer.fit()

    d = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": get_run_id(
            experiment_name=experiment_name, trial_id=results.metrics["trial_id"]
        ),
        "params": results.config["train_loop_config"],
        "metrics": results.metrics_dataframe.to_json(),
    }
    logger.info(json.dumps(d, indent=2))
    return results


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()

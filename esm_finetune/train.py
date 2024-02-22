import ray
import time
import typer
import torch
import torch.nn as nn
import json
import functools
import datetime
import lightning.pytorch as pl

from typing import Optional
from typing_extensions import Annotated
from esm_finetune.data import load_data, CustomPreprocessor
from esm_finetune.utils import collate_fn, get_run_id
from esm_finetune.models import FinetuneESM, ESMLightningModule
from esm_finetune.config import STORAGE_DIR, MLFLOW_TRACKING_URI, logger
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
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    num_devices = config["num_devices"]
    batch_size_per_worker = config["batch_size_per_worker"]
    loss_fn = config["loss_fn"]
    score_name = config["score_name"]
    strategy = config["strategy"]
    verbose = config[True]

    train_ds = ray.train.get_dataset_shard("train")
    val_ds = ray.train.get_dataset_shard("val")
    train_loader = train_ds.iter_torch_batches(
        batch_size=batch_size_per_worker, collate_fn=collate_fn
    )
    val_loader = val_ds.iter_torch_batches(
        batch_size=batch_size_per_worker, collate_fn=collate_fn
    )

    model = FinetuneESM(
        esm_model=esm_model,
        dropout_p=dropout_p,
        num_classes=num_classes,
    )
    lightning_model = ESMLightningModule(
        model=model, learning_rate=lr, loss_fn=loss_fn, score_name=score_name
    )

    callbacks = [RayTrainReportCallback()]

    if score_name == "multilabel_f1_score":
        # save the model with highest f1_score
        callbacks.append(
            ModelCheckpoint(save_top_k=1, mode="max", monitor="val_f1_score")
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
    dataset_loc: Annotated[str, typer.Option("Path to the dataset in parquet format")],
    num_samples: Annotated[
        Optional[int], typer.Option(help="Name for the training experiment")
    ] = None,
    val_size: Annotated[
        float, typer.Option(help="Name for the training experiment")
    ] = 0.25,
    experiment_name: Annotated[
        str, typer.Option(help="Name for the training experiment")
    ] = "unnamed_experiment",
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
    ] = 3,
    loss_function: Annotated[
        str, typer.Option(help="Training strategy")
    ] = "bcewithlogits",
    score_name: Annotated[
        str, typer.Option(help="Training strategy")
    ] = "multilabel_f1_score",
    strategy: Annotated[str, typer.Option(help="Training strategy")] = "auto",
    num_workers: Annotated[
        int, typer.Option("Number of workers to use for training")
    ] = 1,
    verbose: Annotated[
        bool, typer.Option("Number of workers to use for training")
    ] = False,
):
    # configs
    train_loop_config = {
        "esm_model": esm_model,
    }

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
    preprocessor = CustomPreprocessor()
    train_ds = preprocessor.transform(train_ds)
    val_ds = preprocessor.transform(val_ds)
    train_ds = train_ds.materialize()
    val_ds = val_ds.materialize()

    if strategy == "FSDP":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={EsmLayer}
        )

        train_loop_config["strategy"] = RayFSDPStrategy(
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
            auto_wrap_policy=auto_wrap_policy,
            limit_all_gathers=True,
            activation_checkpointing_policy={EsmLayer},
        )

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
        "timestamp": datetime.datetime.now(),
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

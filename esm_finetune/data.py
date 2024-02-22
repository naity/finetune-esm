import ray
import pandas as pd
import numpy as np

from ray.data import Dataset
from typing import Union
from pathlib import Path
from transformers import AutoTokenizer


def load_data(dataset_loc: Union[str, Path], num_samples=None) -> Dataset:
    """
    Loads a dataset in Parquet format from the specified location and performs shuffling and optional sampling.

    Args:
        dataset_loc (str or Path): The path to the dataset file (e.g., Parquet file).
        num_samples (int, optional): The maximum number of samples to load. If None,
            loads the entire dataset.

    Returns:
        The loaded dataset
    """
    ds = ray.data.read_parquet(dataset_loc)
    ds = ds.random_shuffle(seed=0)

    if num_samples is not None:
        # sample at most half dataset
        num_samples = min(num_samples, ds.count() // 2)
        ds = ray.data.from_items(ds.take(num_samples))

    return ds


def tokenize_seqs(
    batch: pd.DataFrame,
    tokenizer: AutoTokenizer,
    targets: np.ndarray,
    max_length: int = 1024,
) -> dict[str, np.ndarray]:
    """
    Tokenizes and encodes a batch of sequences using a provided tokenizer.

    Args:
        batch: A Pandas DataFrame containing the sequences to tokenize.
            The dataframe should have columns "Sequence" and "Index".
            - "Sequence": Strings representing the sequences to tokenize.
            - "Index": Integer indices corresponding to the original samples in the targets array.
        tokenizer (transformers.AutoTokenizer): The tokenizer to use for encoding.
        targets (np.ndarray): A NumPy array of ground-truth labels or targets for the sequences.
        max_length (int, optional): The maximum length to truncate the encoded sequences.
            Defaults to 1024.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the encoded sequences and targets.
            - "input_ids": NumPy array of token IDs for the sequences.
            - "attention_mask": NumPy array of attention masks for the sequences.
            - "targets": NumPy array of ground-truth labels or targets corresponding to the encoded sequences.
    """

    encoded_seqs = tokenizer(
        batch["Sequence"].tolist(),
        padding="longest",
        truncation=True,
        max_length=min(max_length, tokenizer.model_max_length),
        return_tensors="np",
    )
    return dict(
        input_ids=encoded_seqs["input_ids"],
        attention_mask=encoded_seqs["attention_mask"],
        targets=targets[batch["Index"].tolist()],
    )


class CustomPreprocessor:
    """Custom preprocessor class."""

    def __init__(self, esm_model, targets):
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model)
        self.targets = targets

    def transform(self, ds):
        return ds.map_batches(
            tokenize_seqs,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "targets": self.targets,
            },
            batch_format="pandas",
        )

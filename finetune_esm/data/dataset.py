"""Load and normalise the PPB-Affinity dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from finetune_esm import config

COLUMNS = [
    "complex_id", "pdb", "ligand_seqs", "receptor_seqs",
    "pkd", "split", "has_mutation", "source",
]


def kd_to_pkd(kd_molar: np.ndarray) -> np.ndarray:
    """Convert a dissociation constant in molar to pKd = -log10(KD)."""
    return -np.log10(np.asarray(kd_molar, dtype=np.float64))


def download_raw() -> Path:
    """Fetch filtered.csv from the Hub, returning the local path."""
    return Path(
        hf_hub_download(
            repo_id=config.PPB_REPO,
            filename=config.PPB_FILE,
            repo_type="dataset",
        )
    )


def _split_chains(value: str) -> list[str]:
    return [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]


def load_complexes() -> pd.DataFrame:
    """Return the dataset in the normalised schema of COLUMNS."""
    raw = pd.read_csv(download_raw())
    out = pd.DataFrame(
        {
            "complex_id": raw["Complex ID"].astype(str),
            "pdb": raw["PDB"].astype(str),
            "ligand_seqs": raw["Ligand Sequences"].map(_split_chains),
            "receptor_seqs": raw["Receptor Sequences"].map(_split_chains),
            "pkd": kd_to_pkd(raw["KD(M)"].to_numpy()),
            "split": raw["split"].astype(str),
            "has_mutation": raw["Mutations"].notna(),
            "source": raw["Source Data Set"].astype(str),
        }
    )
    return out[COLUMNS]

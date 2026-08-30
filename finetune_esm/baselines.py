"""Non-PLM baseline establishing the floor for this benchmark."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from finetune_esm import metrics
from finetune_esm.data import dataset, splits

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def trivial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Length, chain count, and amino-acid composition. No PLM involved."""
    ligand = df["ligand_seqs"].map("".join)
    receptor = df["receptor_seqs"].map("".join)
    both = ligand + receptor

    feats = pd.DataFrame(
        {
            "ligand_len": ligand.str.len(),
            "receptor_len": receptor.str.len(),
            "n_ligand_chains": df["ligand_seqs"].map(len),
            "n_receptor_chains": df["receptor_seqs"].map(len),
        }
    )
    feats["total_len"] = feats["ligand_len"] + feats["receptor_len"]
    feats["length_ratio"] = feats["ligand_len"] / feats["receptor_len"].clip(lower=1)
    for aa in AMINO_ACIDS:
        feats[f"frac_{aa}"] = both.str.count(aa) / feats["total_len"].clip(lower=1)
    return feats.reset_index(drop=True)


def run_trivial_baseline(seed: int = 0) -> dict[str, float]:
    """Fit on the authors' train split, evaluate on their test split."""
    df = dataset.load_complexes()
    idx = splits.author_split(df)
    feats = trivial_features(df)

    model = HistGradientBoostingRegressor(max_iter=400, random_state=seed)
    model.fit(feats.iloc[idx["train"]], df["pkd"].to_numpy()[idx["train"]])
    pred = model.predict(feats.iloc[idx["test"]])
    return metrics.all_metrics(pred, df["pkd"].to_numpy()[idx["test"]])

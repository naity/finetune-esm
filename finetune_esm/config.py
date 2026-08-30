"""Paths and project-wide constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
CACHE_DIR = ROOT_DIR / "embedding_cache"
RESULTS_DIR = ROOT_DIR / "results"
MLFLOW_DIR = RESULTS_DIR / "mlflow"

MLFLOW_TRACKING_URI = "file://" + str(MLFLOW_DIR)

# Dataset coordinates. CC-BY-NC-SA-4.0; the project is non-commercial.
PPB_REPO = "proteinea/ppb_affinity"
PPB_FILE = "filtered.csv"

# Chains longer than this are truncated. Affects 0.5% of unique chains.
MAX_CHAIN_LENGTH = 1022

# Row counts of the authors' split. Any deviation is a bug.
SPLIT_SIZES = {"train": 6485, "val": 965, "test": 757}

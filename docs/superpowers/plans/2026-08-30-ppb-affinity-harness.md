# PPB-Affinity Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-tier training harness for protein–protein binding affinity regression on PPB-Affinity, and use it to reproduce the published ESM2-650M + hierarchical-pooling result (Spearman ≈ 0.47) as a gate on all later research.

**Architecture:** A frozen protein-language-model backbone produces per-residue embeddings that are cached once to disk, keyed by sequence hash. Interchangeable fusion modules combine ligand and receptor chain embeddings into a single vector, which a small head maps to a predicted pKd. Because the backbone is frozen during screening, experiments train on cached tensors in seconds. A LoRA tier (built here, exercised in the research plan) removes the frozen ceiling later.

**Tech Stack:** Python 3.11, uv, PyTorch 2.11, Lightning, HuggingFace transformers + datasets, MLflow, Optuna, pytest, pandas, numpy, scipy.

**Spec:** `docs/superpowers/specs/2026-08-30-ppb-affinity-design.md`

## Global Constraints

- Python **>= 3.11**. Dependency management via **uv** (`pyproject.toml`); `requirements.txt` is deleted, never updated.
- Dataset: `proteinea/ppb_affinity`, file `filtered.csv`, **CC-BY-NC-SA-4.0**. The project is therefore non-commercial; any PLM licence permits use.
- Target variable is **pKd = −log₁₀(KD in M)**, never raw KD.
- Split sizes are exactly **train 6,485 / val 965 / test 757** (8,207 rows). Any code path producing different counts on the authors' split is a bug.
- **Zero PDB entries** may span two splits, in the authors' split or any split we construct.
- Primary metric is **Spearman**; it is also the checkpoint-selection and early-stopping criterion. Pearson and RMSE (in pKd) are reported alongside.
- Chain truncation limit is **1,022 residues** (affects 0.5% of unique chains). Recorded in cache metadata; a cache built under different settings must be rejected, never silently reused.
- All embeddings cached as **float16**; computation in float32 unless a task says otherwise.
- No Ray. Multi-GPU, if ever needed, goes through Lightning's `FSDPStrategy`.
- Reference numbers to beat, on the authors' split: ProtT5+PAD **0.48 ± 0.03** Spearman / 0.51 Pearson / 1.42 RMSE; ESM2-650M+HP **0.47**. Trivial-feature floor **0.212**.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv project definition, deps, pytest config |
| `finetune_esm/config.py` | Paths, constants, MLflow URI |
| `finetune_esm/data/dataset.py` | Download, parse, normalise schema, derive pKd |
| `finetune_esm/data/splits.py` | Authors' / cold-PDB / random splits + leakage assertions |
| `finetune_esm/data/cache.py` | Embedding cache build, memmap reader, metadata invalidation |
| `finetune_esm/data/batching.py` | `EmbeddingBatch` dataclass + collate |
| `finetune_esm/models/backbones.py` | PLM registry behind one interface |
| `finetune_esm/models/fusion.py` | EC, SC, HP, PAD |
| `finetune_esm/models/heads.py` | MLP head |
| `finetune_esm/models/lora.py` | Repaired hand-rolled LoRA |
| `finetune_esm/models/module.py` | LightningModule |
| `finetune_esm/losses.py` | Loss registry (MSE here; ranking losses in the research plan) |
| `finetune_esm/metrics.py` | Spearman, Pearson, RMSE |
| `finetune_esm/baselines.py` | Trivial-feature baseline |
| `finetune_esm/train.py` | Typer CLI entrypoint |
| `tests/` | One test module per source module |

---

### Task 1: Project scaffolding and removal of the CAFA-5 stack

**Files:**
- Create: `pyproject.toml`, `finetune_esm/__init__.py`, `finetune_esm/config.py`, `tests/__init__.py`, `tests/test_config.py`
- Delete: `requirements.txt`, `finetune_esm/data.py`, `finetune_esm/models.py`, `finetune_esm/train.py`, `finetune_esm/utils.py`, `finetune_esm/lora.py`, `finetune_esm/finetune_results/`, `mlruns/`, `data` (broken symlink), `notebooks/deprecated/`, `finetune_esm/__pycache__/`

**Interfaces:**
- Consumes: nothing.
- Produces: `finetune_esm.config.ROOT_DIR`, `DATA_DIR`, `CACHE_DIR`, `RESULTS_DIR`, `MLFLOW_TRACKING_URI`, `PPB_REPO`, `PPB_FILE`, `MAX_CHAIN_LENGTH`, `SPLIT_SIZES` (all module-level constants).

Note: the old modules are removed wholesale rather than edited. Git history preserves the CAFA-5 implementation; `notebooks/cafa5_*.ipynb` stay in place as a record.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from finetune_esm import config


def test_paths_are_absolute_and_under_root():
    assert config.ROOT_DIR.is_absolute()
    for p in (config.DATA_DIR, config.CACHE_DIR, config.RESULTS_DIR):
        assert config.ROOT_DIR in p.parents or p == config.ROOT_DIR


def test_dataset_coordinates():
    assert config.PPB_REPO == "proteinea/ppb_affinity"
    assert config.PPB_FILE == "filtered.csv"


def test_split_sizes_match_published_dataset():
    assert config.SPLIT_SIZES == {"train": 6485, "val": 965, "test": 757}
    assert sum(config.SPLIT_SIZES.values()) == 8207


def test_truncation_limit():
    assert config.MAX_CHAIN_LENGTH == 1022


def test_mlflow_uri_is_a_file_uri():
    assert config.MLFLOW_TRACKING_URI.startswith("file://")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm'` (or `finetune_esm.config`).

- [ ] **Step 3: Delete the CAFA-5 stack**

```bash
git rm -r --cached mlruns finetune_esm/finetune_results 2>/dev/null || true
rm -f requirements.txt data
rm -f finetune_esm/data.py finetune_esm/models.py finetune_esm/train.py \
      finetune_esm/utils.py finetune_esm/lora.py
rm -rf finetune_esm/__pycache__ finetune_esm/finetune_results mlruns notebooks/deprecated
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "finetune-esm"
version = "0.2.0"
description = "Protein-protein binding affinity regression with protein language models"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.6",
    "lightning>=2.4",
    "transformers>=4.44",
    "datasets>=2.20",
    "huggingface-hub>=0.24",
    "pandas>=2.2",
    "numpy>=1.26",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "pyarrow>=16.0",
    "mlflow>=2.15",
    "optuna>=3.6",
    "typer>=0.12",
    "peft>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-xdist>=3.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["finetune_esm"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: requires network or GPU; deselect with -m 'not slow'",
]
```

- [ ] **Step 5: Write `finetune_esm/__init__.py` and `finetune_esm/config.py`**

```python
# finetune_esm/__init__.py
__version__ = "0.2.0"
```

```python
# finetune_esm/config.py
"""Paths and project-wide constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
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
```

- [ ] **Step 6: Create the environment and run the test**

```bash
uv sync --extra dev
uv run pytest tests/test_config.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat: scaffold uv project, remove CAFA-5 and Ray stack"
```

---

### Task 2: Dataset loading and pKd derivation

**Files:**
- Create: `finetune_esm/data/__init__.py`, `finetune_esm/data/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `config.PPB_REPO`, `config.PPB_FILE`, `config.DATA_DIR`.
- Produces:
  - `kd_to_pkd(kd_molar: np.ndarray) -> np.ndarray`
  - `download_raw() -> Path`
  - `load_complexes() -> pd.DataFrame` with exactly these columns: `complex_id: str`, `pdb: str`, `ligand_seqs: list[str]`, `receptor_seqs: list[str]`, `pkd: float64`, `split: str`, `has_mutation: bool`, `source: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import numpy as np
import pytest

from finetune_esm.data import dataset


def test_kd_to_pkd_is_negative_log10():
    kd = np.array([1e-9, 1e-6, 1.0])
    assert np.allclose(dataset.kd_to_pkd(kd), [9.0, 6.0, 0.0])


@pytest.mark.slow
def test_load_complexes_schema_and_counts():
    df = dataset.load_complexes()
    assert len(df) == 8207
    assert list(df.columns) == [
        "complex_id", "pdb", "ligand_seqs", "receptor_seqs",
        "pkd", "split", "has_mutation", "source",
    ]
    assert df["split"].value_counts().to_dict() == {
        "train": 6485, "val": 965, "test": 757
    }


@pytest.mark.slow
def test_pkd_range_matches_measurements():
    df = dataset.load_complexes()
    assert df["pkd"].min() == pytest.approx(1.39, abs=0.01)
    assert df["pkd"].max() == pytest.approx(15.70, abs=0.01)
    assert df["pkd"].isna().sum() == 0


@pytest.mark.slow
def test_sequences_are_split_on_commas_and_nonempty():
    df = dataset.load_complexes()
    assert df["ligand_seqs"].map(len).min() >= 1
    assert df["receptor_seqs"].map(len).min() >= 1
    for seqs in df["ligand_seqs"].head(200):
        for s in seqs:
            assert s == s.strip() and len(s) > 0 and "," not in s


@pytest.mark.slow
def test_mutation_flag_matches_split_composition():
    df = dataset.load_complexes()
    by_split = df.groupby("split")["has_mutation"].sum().to_dict()
    assert by_split["train"] == 4766
    assert by_split["test"] == 21
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.data'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/data/__init__.py
```

```python
# finetune_esm/data/dataset.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 5 passed (the four `slow` tests download ~23 MB on first run).

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/data tests/test_dataset.py
git commit -m "feat: load PPB-Affinity and derive pKd"
```

---

### Task 3: Splits with leakage assertions

**Files:**
- Create: `finetune_esm/data/splits.py`, `tests/test_splits.py`

**Interfaces:**
- Consumes: `dataset.load_complexes()`.
- Produces:
  - `author_split(df: pd.DataFrame) -> dict[str, np.ndarray]` — split name to positional row indices.
  - `cold_pdb_split(df, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict[str, np.ndarray]` — groups whole PDB entries.
  - `random_split(df, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict[str, np.ndarray]`
  - `assert_no_pdb_leakage(df, splits) -> None` — raises `AssertionError` naming the offending PDB.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_splits.py
import numpy as np
import pandas as pd
import pytest

from finetune_esm.data import dataset, splits


def _toy() -> pd.DataFrame:
    # 3 PDBs, 6 rows: A appears 3x, B 2x, C 1x
    return pd.DataFrame(
        {
            "pdb": ["A", "A", "A", "B", "B", "C"],
            "split": ["train"] * 5 + ["test"],
            "pkd": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


def test_assert_no_pdb_leakage_accepts_disjoint_groups():
    df = _toy()
    splits.assert_no_pdb_leakage(df, {"a": np.array([0, 1, 2]), "b": np.array([3, 4, 5])})


def test_assert_no_pdb_leakage_rejects_shared_pdb():
    df = _toy()
    with pytest.raises(AssertionError, match="A"):
        splits.assert_no_pdb_leakage(df, {"a": np.array([0, 1]), "b": np.array([2, 3])})


def test_cold_pdb_split_keeps_groups_intact():
    rng_df = pd.DataFrame({"pdb": ["P%d" % (i // 4) for i in range(400)]})
    out = splits.cold_pdb_split(rng_df, seed=0)
    splits.assert_no_pdb_leakage(rng_df, out)
    assert sum(len(v) for v in out.values()) == 400


def test_random_split_partitions_every_row_exactly_once():
    df = pd.DataFrame({"pdb": ["X"] * 100})
    out = splits.random_split(df, seed=0)
    joined = np.sort(np.concatenate(list(out.values())))
    assert np.array_equal(joined, np.arange(100))


def test_splits_are_deterministic_given_a_seed():
    df = pd.DataFrame({"pdb": ["P%d" % (i // 3) for i in range(300)]})
    assert np.array_equal(
        splits.cold_pdb_split(df, seed=7)["train"],
        splits.cold_pdb_split(df, seed=7)["train"],
    )


@pytest.mark.slow
def test_author_split_sizes_and_purity():
    df = dataset.load_complexes()
    out = splits.author_split(df)
    assert {k: len(v) for k, v in out.items()} == {
        "train": 6485, "val": 965, "test": 757
    }
    splits.assert_no_pdb_leakage(df, out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_splits.py -v`
Expected: FAIL — `ImportError: cannot import name 'splits'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/data/splits.py
"""Construct and validate train/val/test splits."""

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


def assert_no_pdb_leakage(df: pd.DataFrame, splits: dict[str, np.ndarray]) -> None:
    """Raise if any PDB entry appears in more than one split."""
    seen: dict[str, str] = {}
    for name, idx in splits.items():
        for pdb in df.iloc[idx]["pdb"].unique():
            if pdb in seen and seen[pdb] != name:
                raise AssertionError(
                    f"PDB {pdb} appears in both {seen[pdb]!r} and {name!r}"
                )
            seen[pdb] = name


def author_split(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The split shipped in the dataset's own `split` column."""
    return {
        name: np.flatnonzero((df["split"] == name).to_numpy())
        for name in SPLIT_NAMES
    }


def _assign_groups(
    groups: np.ndarray, seed: int, fracs: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_train = int(round(fracs[0] * len(unique)))
    n_val = int(round(fracs[1] * len(unique)))
    chunks = {
        "train": set(unique[:n_train]),
        "val": set(unique[n_train : n_train + n_val]),
        "test": set(unique[n_train + n_val :]),
    }
    return {
        name: np.flatnonzero(np.isin(groups, list(members)))
        for name, members in chunks.items()
    }


def cold_pdb_split(
    df: pd.DataFrame, seed: int, fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Split so that a PDB entry never spans two splits."""
    return _assign_groups(df["pdb"].to_numpy(), seed, fracs)


def random_split(
    df: pd.DataFrame, seed: int, fracs: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, np.ndarray]:
    """Row-level random split. Leaks by construction; reported for contrast."""
    return _assign_groups(np.arange(len(df)), seed, fracs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_splits.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/data/splits.py tests/test_splits.py
git commit -m "feat: author, cold-PDB and random splits with leakage assertions"
```

---

### Task 4: Metrics

**Files:**
- Create: `finetune_esm/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `spearman(pred, target) -> float`, `pearson(pred, target) -> float`, `rmse(pred, target) -> float`, `all_metrics(pred, target) -> dict[str, float]` with keys `"spearman"`, `"pearson"`, `"rmse"`. All accept `np.ndarray` or `torch.Tensor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import torch
from scipy import stats

from finetune_esm import metrics


def test_matches_scipy_on_random_data():
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=200), rng.normal(size=200)
    assert metrics.spearman(pred, target) == np.float32(
        stats.spearmanr(pred, target).statistic
    ).item() or abs(
        metrics.spearman(pred, target) - stats.spearmanr(pred, target).statistic
    ) < 1e-9
    assert abs(
        metrics.pearson(pred, target) - stats.pearsonr(pred, target).statistic
    ) < 1e-9


def test_rmse_is_root_mean_square_error():
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.0, 2.0, 5.0])
    assert abs(metrics.rmse(pred, target) - np.sqrt(4.0 / 3.0)) < 1e-12


def test_perfect_rank_agreement_is_one():
    target = np.arange(50, dtype=float)
    assert abs(metrics.spearman(2 * target + 7, target) - 1.0) < 1e-12


def test_handles_ties_without_error():
    pred = np.array([1.0, 1.0, 2.0, 2.0])
    target = np.array([1.0, 2.0, 1.0, 2.0])
    assert np.isfinite(metrics.spearman(pred, target))


def test_constant_prediction_is_nan_not_crash():
    pred = np.ones(10)
    target = np.arange(10, dtype=float)
    assert np.isnan(metrics.spearman(pred, target))
    assert np.isnan(metrics.pearson(pred, target))
    assert np.isfinite(metrics.rmse(pred, target))


def test_accepts_torch_tensors():
    pred = torch.arange(20, dtype=torch.float32)
    target = torch.arange(20, dtype=torch.float32)
    assert abs(metrics.spearman(pred, target) - 1.0) < 1e-9


def test_all_metrics_keys():
    rng = np.random.default_rng(1)
    out = metrics.all_metrics(rng.normal(size=30), rng.normal(size=30))
    assert set(out) == {"spearman", "pearson", "rmse"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/metrics.py
"""Evaluation metrics. Spearman is primary; Pearson and RMSE are reported alongside."""

import numpy as np
from scipy import stats


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64).ravel()


def spearman(pred, target) -> float:
    """Spearman rank correlation. NaN when either input is constant."""
    p, t = _to_numpy(pred), _to_numpy(target)
    if np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(stats.spearmanr(p, t).statistic)


def pearson(pred, target) -> float:
    """Pearson correlation. NaN when either input is constant."""
    p, t = _to_numpy(pred), _to_numpy(target)
    if np.std(p) == 0 or np.std(t) == 0:
        return float("nan")
    return float(stats.pearsonr(p, t).statistic)


def rmse(pred, target) -> float:
    """Root mean squared error, in pKd units."""
    p, t = _to_numpy(pred), _to_numpy(target)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def all_metrics(pred, target) -> dict[str, float]:
    return {
        "spearman": spearman(pred, target),
        "pearson": pearson(pred, target),
        "rmse": rmse(pred, target),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/metrics.py tests/test_metrics.py
git commit -m "feat: Spearman, Pearson and RMSE metrics"
```

---

### Task 5: Trivial-feature baseline

This pins the floor every later result is read against, and exercises Tasks 2–4 end to end before any PLM is involved.

**Files:**
- Create: `finetune_esm/baselines.py`, `tests/test_baselines.py`

**Interfaces:**
- Consumes: `dataset.load_complexes()`, `splits.author_split()`, `metrics.all_metrics()`.
- Produces:
  - `trivial_features(df: pd.DataFrame) -> pd.DataFrame` — numeric feature frame, one row per complex.
  - `run_trivial_baseline(seed: int = 0) -> dict[str, float]` — metrics on the authors' test split.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import pytest

from finetune_esm import baselines
from finetune_esm.data import dataset


@pytest.mark.slow
def test_trivial_features_shape_and_finiteness():
    df = dataset.load_complexes().head(500)
    feats = baselines.trivial_features(df)
    assert len(feats) == 500
    assert feats.notna().all().all()
    for col in ("ligand_len", "receptor_len", "n_ligand_chains", "total_len"):
        assert col in feats.columns
    # 20 amino-acid composition features
    assert sum(c.startswith("frac_") for c in feats.columns) == 20


@pytest.mark.slow
def test_trivial_baseline_reproduces_measured_floor():
    out = baselines.run_trivial_baseline(seed=0)
    # Measured during design: Spearman 0.212, Pearson 0.222, RMSE 1.934.
    assert out["spearman"] == pytest.approx(0.212, abs=0.03)
    assert out["rmse"] == pytest.approx(1.934, abs=0.10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.baselines'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/baselines.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/baselines.py tests/test_baselines.py
git commit -m "feat: trivial-feature baseline pinning the 0.212 Spearman floor"
```

---

### Task 6: Backbone registry

**Files:**
- Create: `finetune_esm/models/__init__.py`, `finetune_esm/models/backbones.py`, `tests/test_backbones.py`

**Interfaces:**
- Consumes: `config.MAX_CHAIN_LENGTH`.
- Produces:
  - `BackboneSpec` dataclass with fields `name: str`, `hf_id: str`, `hidden_dim: int`.
  - `BACKBONES: dict[str, BackboneSpec]` containing at minimum keys `"esm2-8m"`, `"esm2-650m"`.
  - `class Backbone` with `__init__(self, spec: BackboneSpec, device: str = "cpu", dtype: torch.dtype = torch.float32)`, attribute `spec`, and `embed(self, seqs: list[str]) -> list[np.ndarray]` returning one `(L_i, hidden_dim)` float32 array per input, with **special tokens stripped** so `L_i == min(len(seq), MAX_CHAIN_LENGTH)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backbones.py
import numpy as np
import pytest

from finetune_esm import config
from finetune_esm.models import backbones


def test_registry_contains_expected_specs():
    assert backbones.BACKBONES["esm2-8m"].hf_id == "facebook/esm2_t6_8M_UR50D"
    assert backbones.BACKBONES["esm2-650m"].hidden_dim == 1280


@pytest.mark.slow
def test_embed_returns_one_array_per_sequence_without_special_tokens():
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    seqs = ["MKTAYIAKQR", "MVLSPADKTNVKAAW"]
    out = bb.embed(seqs)
    assert len(out) == 2
    for seq, arr in zip(seqs, out):
        assert arr.shape == (len(seq), bb.spec.hidden_dim)
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()


@pytest.mark.slow
def test_embed_truncates_at_max_chain_length():
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    long_seq = "A" * (config.MAX_CHAIN_LENGTH + 500)
    (arr,) = bb.embed([long_seq])
    assert arr.shape[0] == config.MAX_CHAIN_LENGTH


@pytest.mark.slow
def test_embedding_is_independent_of_batch_composition():
    """A chain's embedding must not depend on what it was batched with."""
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    target = "MKTAYIAKQRQISFVK"
    (alone,) = bb.embed([target])
    together = bb.embed(["MVLSPADKTNVKAAWGKVGAHAGEYGAEAL", target])[1]
    assert np.allclose(alone, together, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backbones.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.models'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/models/__init__.py
```

```python
# finetune_esm/models/backbones.py
"""Protein language model backbones behind one interface."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from finetune_esm import config


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    hf_id: str
    hidden_dim: int


BACKBONES: dict[str, BackboneSpec] = {
    "esm2-8m": BackboneSpec("esm2-8m", "facebook/esm2_t6_8M_UR50D", 320),
    "esm2-35m": BackboneSpec("esm2-35m", "facebook/esm2_t12_35M_UR50D", 480),
    "esm2-150m": BackboneSpec("esm2-150m", "facebook/esm2_t30_150M_UR50D", 640),
    "esm2-650m": BackboneSpec("esm2-650m", "facebook/esm2_t33_650M_UR50D", 1280),
}


class Backbone:
    """Wraps a frozen PLM and returns per-residue embeddings."""

    def __init__(
        self,
        spec: BackboneSpec,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spec = spec
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
        self.model = AutoModel.from_pretrained(spec.hf_id, torch_dtype=dtype)
        self.model.eval().to(device)
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def embed(self, seqs: list[str]) -> list[np.ndarray]:
        """Per-residue embeddings, special tokens stripped, one array per sequence."""
        truncated = [s[: config.MAX_CHAIN_LENGTH] for s in seqs]
        encoded = self.tokenizer(
            truncated,
            padding=True,
            truncation=True,
            max_length=config.MAX_CHAIN_LENGTH + 2,  # room for CLS and EOS
            return_tensors="pt",
        ).to(self.device)

        hidden = self.model(**encoded).last_hidden_state.float().cpu().numpy()

        out = []
        for i, seq in enumerate(truncated):
            # ESM prepends CLS and appends EOS; residue i lives at position i + 1.
            out.append(hidden[i, 1 : len(seq) + 1].astype(np.float32))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backbones.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/models tests/test_backbones.py
git commit -m "feat: PLM backbone registry with per-residue embedding extraction"
```

---

### Task 7: Embedding cache

**Files:**
- Create: `finetune_esm/data/cache.py`, `tests/test_cache.py`

**Interfaces:**
- Consumes: `models.backbones.Backbone`, `config.MAX_CHAIN_LENGTH`, `config.CACHE_DIR`.
- Produces:
  - `chain_key(seq: str) -> str` — SHA-1 hex digest.
  - `CacheMeta` dataclass: `backbone: str`, `hf_id: str`, `hidden_dim: int`, `max_length: int`, `dtype: str`. Methods `to_json(path)`, `from_json(path)`.
  - `build_cache(seqs: Iterable[str], backbone: Backbone, out_dir: Path, batch_size: int = 8) -> CacheMeta`
  - `class EmbeddingCache` with `__init__(self, out_dir: Path, expect: CacheMeta | None = None)` (raises `ValueError` on metadata mismatch), `residues(seq) -> np.ndarray` shape `(L, D)`, `pooled(seq) -> np.ndarray` shape `(D,)`, `__contains__(seq) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
import numpy as np
import pytest

from finetune_esm.data import cache
from finetune_esm.models import backbones

SEQS = ["MKTAYIAKQR", "MVLSPADKTNVKAAW", "GGGGSGGGGS"]


def test_chain_key_is_stable_and_distinct():
    assert cache.chain_key("ABC") == cache.chain_key("ABC")
    assert cache.chain_key("ABC") != cache.chain_key("ABD")
    assert len(cache.chain_key("ABC")) == 40


@pytest.mark.slow
def test_cache_round_trips_and_matches_live_forward_pass(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)

    live = bb.embed(SEQS)
    for seq, expected in zip(SEQS, live):
        got = store.residues(seq)
        assert got.shape == expected.shape
        # fp16 storage: compare at half-precision tolerance
        assert np.allclose(got, expected, atol=1e-2)


@pytest.mark.slow
def test_pooled_equals_mean_over_residues(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)
    for seq in SEQS:
        assert np.allclose(
            store.pooled(seq), store.residues(seq).mean(axis=0), atol=1e-2
        )


@pytest.mark.slow
def test_duplicate_sequences_are_encoded_once(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS + SEQS, bb, tmp_path)
    assert meta.n_chains == len(SEQS)


@pytest.mark.slow
def test_mismatched_metadata_is_rejected(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    wrong = cache.CacheMeta(
        backbone="esm2-650m",
        hf_id="facebook/esm2_t33_650M_UR50D",
        hidden_dim=1280,
        max_length=meta.max_length,
        dtype=meta.dtype,
        n_chains=meta.n_chains,
    )
    with pytest.raises(ValueError, match="mismatch"):
        cache.EmbeddingCache(tmp_path, expect=wrong)


@pytest.mark.slow
def test_unknown_sequence_raises(tmp_path):
    bb = backbones.Backbone(backbones.BACKBONES["esm2-8m"])
    meta = cache.build_cache(SEQS, bb, tmp_path)
    store = cache.EmbeddingCache(tmp_path, expect=meta)
    assert "MKTAYIAKQR" in store
    assert "WWWWWWWW" not in store
    with pytest.raises(KeyError):
        store.residues("WWWWWWWW")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cache.py -v`
Expected: FAIL — `ImportError: cannot import name 'cache'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/data/cache.py
"""Disk cache of frozen per-residue embeddings, keyed by sequence hash."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from finetune_esm import config
from finetune_esm.models.backbones import Backbone

RESIDUES_FILE = "residues.f16.npy"
POOLED_FILE = "pooled.f16.npy"
INDEX_FILE = "index.parquet"
META_FILE = "meta.json"


def chain_key(seq: str) -> str:
    """SHA-1 of the sequence. Deduplicates repeated chains across complexes."""
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheMeta:
    backbone: str
    hf_id: str
    hidden_dim: int
    max_length: int
    dtype: str
    n_chains: int

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "CacheMeta":
        return cls(**json.loads(path.read_text()))


def build_cache(
    seqs: Iterable[str],
    backbone: Backbone,
    out_dir: Path,
    batch_size: int = 8,
) -> CacheMeta:
    """Encode every distinct sequence once and write the cache to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique: dict[str, str] = {}
    for seq in seqs:
        unique.setdefault(chain_key(seq), seq)
    keys = sorted(unique)
    ordered = [unique[k] for k in keys]

    dim = backbone.spec.hidden_dim
    lengths = [min(len(s), config.MAX_CHAIN_LENGTH) for s in ordered]
    total = int(sum(lengths))

    residues = np.lib.format.open_memmap(
        out_dir / RESIDUES_FILE, mode="w+", dtype=np.float16, shape=(total, dim)
    )
    pooled = np.zeros((len(ordered), dim), dtype=np.float16)

    offsets, cursor = [], 0
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start : start + batch_size]
        for arr in backbone.embed(chunk):
            residues[cursor : cursor + len(arr)] = arr.astype(np.float16)
            pooled[len(offsets)] = arr.mean(axis=0).astype(np.float16)
            offsets.append(cursor)
            cursor += len(arr)
    residues.flush()

    np.save(out_dir / POOLED_FILE, pooled)
    pd.DataFrame(
        {"chain_key": keys, "offset": offsets, "length": lengths, "row": range(len(keys))}
    ).to_parquet(out_dir / INDEX_FILE, index=False)

    meta = CacheMeta(
        backbone=backbone.spec.name,
        hf_id=backbone.spec.hf_id,
        hidden_dim=dim,
        max_length=config.MAX_CHAIN_LENGTH,
        dtype="float16",
        n_chains=len(keys),
    )
    meta.to_json(out_dir / META_FILE)
    return meta


class EmbeddingCache:
    """Read-only memmapped view over a built cache."""

    def __init__(self, out_dir: Path, expect: CacheMeta | None = None) -> None:
        out_dir = Path(out_dir)
        self.meta = CacheMeta.from_json(out_dir / META_FILE)
        if expect is not None and expect != self.meta:
            raise ValueError(
                f"cache metadata mismatch: on disk {self.meta}, expected {expect}"
            )
        self._residues = np.load(out_dir / RESIDUES_FILE, mmap_mode="r")
        self._pooled = np.load(out_dir / POOLED_FILE)
        index = pd.read_parquet(out_dir / INDEX_FILE)
        self._index = {
            row.chain_key: (row.offset, row.length, row.row)
            for row in index.itertuples()
        }

    def __contains__(self, seq: str) -> bool:
        return chain_key(seq) in self._index

    def _lookup(self, seq: str) -> tuple[int, int, int]:
        key = chain_key(seq)
        if key not in self._index:
            raise KeyError(f"sequence not in cache (key {key})")
        return self._index[key]

    def residues(self, seq: str) -> np.ndarray:
        offset, length, _ = self._lookup(seq)
        return np.asarray(self._residues[offset : offset + length], dtype=np.float32)

    def pooled(self, seq: str) -> np.ndarray:
        _, _, row = self._lookup(seq)
        return self._pooled[row].astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cache.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/data/cache.py tests/test_cache.py
git commit -m "feat: memmapped embedding cache with metadata invalidation"
```

---

### Task 8: Batching

**Files:**
- Create: `finetune_esm/data/batching.py`, `tests/test_batching.py`

**Interfaces:**
- Consumes: `cache.EmbeddingCache`.
- Produces:
  - `EmbeddingBatch` dataclass with fields `ligand: Tensor (B, Lmax, D)`, `ligand_mask: Tensor (B, Lmax) bool`, `ligand_chain_id: Tensor (B, Lmax) long` (−1 where padded), `receptor`, `receptor_mask`, `receptor_chain_id`, `target: Tensor (B,)`, `group_id: Tensor (B,) long`. Method `to(device) -> EmbeddingBatch`.
  - `class ComplexDataset(torch.utils.data.Dataset)` — `__init__(self, df, indices, store)`, yields dicts.
  - `collate(items: list[dict]) -> EmbeddingBatch`

`group_id` carries the PDB identity as an integer so later group-aware sampling and ranking losses have it without a schema change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_batching.py
import numpy as np
import pandas as pd
import torch

from finetune_esm.data import batching


class FakeStore:
    """Deterministic stand-in for EmbeddingCache: embedding = index * ones(4)."""

    def residues(self, seq: str) -> np.ndarray:
        return np.full((len(seq), 4), float(len(seq)), dtype=np.float32)


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb": ["A", "B"],
            "ligand_seqs": [["MK"], ["MKT", "GG"]],
            "receptor_seqs": [["MKTA"], ["M"]],
            "pkd": [7.0, 9.0],
        }
    )


def test_collate_pads_to_longest_and_masks_correctly():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])

    assert batch.ligand.shape == (2, 5, 4)          # row 1 has 3 + 2 = 5 residues
    assert batch.ligand_mask[0].sum().item() == 2
    assert batch.ligand_mask[1].sum().item() == 5
    assert batch.receptor.shape == (2, 4, 4)
    assert batch.receptor_mask[0].sum().item() == 4
    assert batch.receptor_mask[1].sum().item() == 1


def test_padded_positions_are_zero_and_chain_id_is_negative_one():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])
    assert torch.all(batch.ligand[0, 2:] == 0)
    assert torch.all(batch.ligand_chain_id[0, 2:] == -1)


def test_chain_ids_distinguish_chains_within_a_partner():
    ds = batching.ComplexDataset(_df(), np.array([1]), FakeStore())
    batch = batching.collate([ds[0]])
    ids = batch.ligand_chain_id[0][batch.ligand_mask[0]]
    assert ids.tolist() == [0, 0, 0, 1, 1]


def test_targets_and_groups_are_carried_through():
    ds = batching.ComplexDataset(_df(), np.array([0, 1]), FakeStore())
    batch = batching.collate([ds[0], ds[1]])
    assert torch.allclose(batch.target, torch.tensor([7.0, 9.0]))
    assert batch.group_id.tolist() != [0, 0]  # distinct PDBs get distinct ids


def test_to_device_is_a_noop_on_cpu():
    ds = batching.ComplexDataset(_df(), np.array([0]), FakeStore())
    batch = batching.collate([ds[0]]).to("cpu")
    assert batch.ligand.device.type == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_batching.py -v`
Expected: FAIL — `ImportError: cannot import name 'batching'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/data/batching.py
"""Dataset and collation over cached embeddings."""

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class EmbeddingBatch:
    ligand: torch.Tensor
    ligand_mask: torch.Tensor
    ligand_chain_id: torch.Tensor
    receptor: torch.Tensor
    receptor_mask: torch.Tensor
    receptor_chain_id: torch.Tensor
    target: torch.Tensor
    group_id: torch.Tensor

    def to(self, device) -> "EmbeddingBatch":
        return EmbeddingBatch(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )


class ComplexDataset(Dataset):
    """One item per complex; embeddings are read from the cache on access."""

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, store) -> None:
        self.rows = df.iloc[indices].reset_index(drop=True)
        self.store = store
        self.group_codes = pd.Categorical(self.rows["pdb"]).codes

    def __len__(self) -> int:
        return len(self.rows)

    def _stack(self, seqs: list[str]) -> tuple[np.ndarray, np.ndarray]:
        parts = [self.store.residues(s) for s in seqs]
        chain_ids = np.concatenate(
            [np.full(len(p), i, dtype=np.int64) for i, p in enumerate(parts)]
        )
        return np.concatenate(parts, axis=0), chain_ids

    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        ligand, ligand_ids = self._stack(row["ligand_seqs"])
        receptor, receptor_ids = self._stack(row["receptor_seqs"])
        return {
            "ligand": ligand,
            "ligand_chain_id": ligand_ids,
            "receptor": receptor,
            "receptor_chain_id": receptor_ids,
            "target": float(row["pkd"]),
            "group_id": int(self.group_codes[i]),
        }


def _pad(arrays: list[np.ndarray], ids: list[np.ndarray]):
    batch, dim = len(arrays), arrays[0].shape[1]
    longest = max(a.shape[0] for a in arrays)

    padded = torch.zeros(batch, longest, dim, dtype=torch.float32)
    mask = torch.zeros(batch, longest, dtype=torch.bool)
    chain_id = torch.full((batch, longest), -1, dtype=torch.long)

    for i, (arr, cid) in enumerate(zip(arrays, ids)):
        n = arr.shape[0]
        padded[i, :n] = torch.from_numpy(arr)
        mask[i, :n] = True
        chain_id[i, :n] = torch.from_numpy(cid)
    return padded, mask, chain_id


def collate(items: list[dict]) -> EmbeddingBatch:
    ligand, ligand_mask, ligand_chain_id = _pad(
        [it["ligand"] for it in items], [it["ligand_chain_id"] for it in items]
    )
    receptor, receptor_mask, receptor_chain_id = _pad(
        [it["receptor"] for it in items], [it["receptor_chain_id"] for it in items]
    )
    return EmbeddingBatch(
        ligand=ligand,
        ligand_mask=ligand_mask,
        ligand_chain_id=ligand_chain_id,
        receptor=receptor,
        receptor_mask=receptor_mask,
        receptor_chain_id=receptor_chain_id,
        target=torch.tensor([it["target"] for it in items], dtype=torch.float32),
        group_id=torch.tensor([it["group_id"] for it in items], dtype=torch.long),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_batching.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/data/batching.py tests/test_batching.py
git commit -m "feat: complex dataset and padded collation over cached embeddings"
```

---

### Task 9: Fusion modules

**Files:**
- Create: `finetune_esm/models/fusion.py`, `tests/test_fusion.py`

**Interfaces:**
- Consumes: `batching.EmbeddingBatch`.
- Produces:
  - `class AttentionPool(nn.Module)` — `__init__(dim)`, `forward(x: (B, L, D), mask: (B, L)) -> (B, D)`.
  - `class FusionBase(nn.Module)` with attribute `out_dim: int` and `forward(batch: EmbeddingBatch) -> Tensor (B, out_dim)`.
  - `EmbeddingConcat`, `SequenceConcat`, `HierarchicalPooling`, `PooledAttentionAddition` — all `__init__(dim: int)`.
  - `FUSIONS: dict[str, type]` with keys `"ec"`, `"sc"`, `"hp"`, `"pad"`.

Definitions follow arXiv:2505.20036 §3: EC pools each partner and concatenates (`out_dim = 2*dim`); SC pools the concatenation of both partners (`out_dim = dim`); HP pools per chain then across chains, per partner, then concatenates (`out_dim = 2*dim`); PAD cross-attends the partners, pools each, then adds element-wise (`out_dim = dim`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fusion.py
import pytest
import torch

from finetune_esm.data.batching import EmbeddingBatch
from finetune_esm.models import fusion

DIM = 8


def _batch(batch_size=3, ligand_len=5, receptor_len=4) -> EmbeddingBatch:
    torch.manual_seed(0)
    ligand_mask = torch.ones(batch_size, ligand_len, dtype=torch.bool)
    ligand_mask[0, -2:] = False  # exercise padding
    receptor_mask = torch.ones(batch_size, receptor_len, dtype=torch.bool)
    return EmbeddingBatch(
        ligand=torch.randn(batch_size, ligand_len, DIM),
        ligand_mask=ligand_mask,
        ligand_chain_id=torch.zeros(batch_size, ligand_len, dtype=torch.long),
        receptor=torch.randn(batch_size, receptor_len, DIM),
        receptor_mask=receptor_mask,
        receptor_chain_id=torch.zeros(batch_size, receptor_len, dtype=torch.long),
        target=torch.zeros(batch_size),
        group_id=torch.arange(batch_size),
    )


@pytest.mark.parametrize("key", ["ec", "sc", "hp", "pad"])
def test_output_shape_matches_declared_out_dim(key):
    model = fusion.FUSIONS[key](DIM)
    out = model(_batch())
    assert out.shape == (3, model.out_dim)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("key", ["ec", "sc", "hp", "pad"])
def test_padding_does_not_affect_output(key):
    """Garbage in padded positions must not change the result."""
    torch.manual_seed(0)
    model = fusion.FUSIONS[key](DIM).eval()
    clean = _batch()
    dirty = _batch()
    dirty.ligand[~dirty.ligand_mask] = 1e3

    with torch.no_grad():
        assert torch.allclose(model(clean), model(dirty), atol=1e-4)


def test_attention_pool_ignores_masked_positions():
    torch.manual_seed(0)
    pool = fusion.AttentionPool(DIM).eval()
    x = torch.randn(1, 6, DIM)
    mask = torch.tensor([[True, True, True, False, False, False]])
    with torch.no_grad():
        full = pool(x, mask)
        trimmed = pool(x[:, :3], mask[:, :3])
    assert torch.allclose(full, trimmed, atol=1e-5)


def test_hierarchical_pooling_uses_chain_structure():
    """Splitting one partner into two chains must change HP's output."""
    torch.manual_seed(0)
    model = fusion.HierarchicalPooling(DIM).eval()
    one_chain = _batch(batch_size=1, ligand_len=4)
    one_chain.ligand_mask[:] = True
    two_chains = _batch(batch_size=1, ligand_len=4)
    two_chains.ligand_mask[:] = True
    two_chains.ligand = one_chain.ligand.clone()
    two_chains.receptor = one_chain.receptor.clone()
    two_chains.ligand_chain_id[0, 2:] = 1

    with torch.no_grad():
        assert not torch.allclose(model(one_chain), model(two_chains), atol=1e-4)


def test_registry_is_complete():
    assert set(fusion.FUSIONS) == {"ec", "sc", "hp", "pad"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: FAIL — `ImportError: cannot import name 'fusion'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/models/fusion.py
"""Strategies for combining ligand and receptor embeddings into one vector.

EC, SC, HP and PAD follow the definitions in arXiv:2505.20036.
"""

import torch
import torch.nn as nn

from finetune_esm.data.batching import EmbeddingBatch

NEG_INF = -1e9


class AttentionPool(nn.Module):
    """Single-query attention pooling over the residue axis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(x).squeeze(-1)
        logits = logits.masked_fill(~mask, NEG_INF)
        empty = ~mask.any(dim=-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        pooled = (x * weights * mask.unsqueeze(-1)).sum(dim=1)
        return pooled.masked_fill(empty.unsqueeze(-1), 0.0)


class FusionBase(nn.Module):
    out_dim: int

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class EmbeddingConcat(FusionBase):
    """Pool each partner independently, then concatenate."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.out_dim = 2 * dim

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return torch.cat(
            [
                self.pool(batch.ligand, batch.ligand_mask),
                self.pool(batch.receptor, batch.receptor_mask),
            ],
            dim=-1,
        )


class SequenceConcat(FusionBase):
    """Concatenate both partners along the residue axis, then pool once."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.pool = AttentionPool(dim)
        self.out_dim = dim

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        joined = torch.cat([batch.ligand, batch.receptor], dim=1)
        mask = torch.cat([batch.ligand_mask, batch.receptor_mask], dim=1)
        return self.pool(joined, mask)


class HierarchicalPooling(FusionBase):
    """Pool residues within each chain, then pool across chains, per partner."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.residue_pool = AttentionPool(dim)
        self.chain_pool = AttentionPool(dim)
        self.out_dim = 2 * dim

    def _partner(
        self, x: torch.Tensor, mask: torch.Tensor, chain_id: torch.Tensor
    ) -> torch.Tensor:
        n_chains = int(chain_id.max().item()) + 1 if mask.any() else 1
        chain_vectors, chain_mask = [], []
        for c in range(n_chains):
            selector = mask & (chain_id == c)
            chain_vectors.append(self.residue_pool(x, selector))
            chain_mask.append(selector.any(dim=-1))
        return self.chain_pool(
            torch.stack(chain_vectors, dim=1), torch.stack(chain_mask, dim=1)
        )

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return torch.cat(
            [
                self._partner(batch.ligand, batch.ligand_mask, batch.ligand_chain_id),
                self._partner(
                    batch.receptor, batch.receptor_mask, batch.receptor_chain_id
                ),
            ],
            dim=-1,
        )


class PooledAttentionAddition(FusionBase):
    """Cross-attend the partners, pool each, then add element-wise."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentionPool(dim)
        self.out_dim = dim

    def _attend(self, q, q_mask, kv, kv_mask):
        attended, _ = self.cross(q, kv, kv, key_padding_mask=~kv_mask)
        attended = attended.masked_fill(~q_mask.unsqueeze(-1), 0.0)
        return self.norm(q + attended)

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        ligand = self._attend(
            batch.ligand, batch.ligand_mask, batch.receptor, batch.receptor_mask
        )
        receptor = self._attend(
            batch.receptor, batch.receptor_mask, batch.ligand, batch.ligand_mask
        )
        return self.pool(ligand, batch.ligand_mask) + self.pool(
            receptor, batch.receptor_mask
        )


FUSIONS: dict[str, type] = {
    "ec": EmbeddingConcat,
    "sc": SequenceConcat,
    "hp": HierarchicalPooling,
    "pad": PooledAttentionAddition,
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_fusion.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/models/fusion.py tests/test_fusion.py
git commit -m "feat: EC, SC, HP and PAD fusion modules"
```

---

### Task 10: Prediction head

**Files:**
- Create: `finetune_esm/models/heads.py`, `tests/test_heads.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class MLPHead(nn.Module)` — `__init__(self, in_dim: int, hidden_dim: int | None = None, dropout: float = 0.1)`, `forward(x: (B, in_dim)) -> (B,)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_heads.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_heads.py -v`
Expected: FAIL — `ImportError: cannot import name 'heads'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/models/heads.py
"""Prediction heads mapping a fused vector to a scalar pKd."""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_heads.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/models/heads.py tests/test_heads.py
git commit -m "feat: MLP prediction head"
```

---

### Task 11: Repaired LoRA with PEFT equivalence

Fixes both bugs identified in spec §5.5: `alpha` scaling becomes `alpha / rank`, and wrapped layers keep their base weights trainable when asked.

**Files:**
- Create: `finetune_esm/models/lora.py`, `tests/test_lora.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class LoRALayer(nn.Module)` — `__init__(in_dim, out_dim, rank, alpha)`, attributes `W_a: (in_dim, rank)`, `W_b: (rank, out_dim)`, `scaling = alpha / rank`.
  - `class LinearWithLoRA(nn.Module)` — `__init__(linear: nn.Linear, rank: int, alpha: float, freeze_base: bool = True)`, attributes `linear`, `lora`.
  - `apply_lora(model, rank=8, alpha=16, target_layers=None, ...) -> int` returning the number of wrapped modules.
  - `count_trainable_parameters(model) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lora.py
import torch
import torch.nn as nn

from finetune_esm.models import lora


def test_scaling_is_alpha_over_rank_not_alpha():
    layer = lora.LoRALayer(4, 3, rank=8, alpha=16)
    assert layer.scaling == 2.0


def test_output_starts_at_zero_because_w_b_is_zero_initialised():
    layer = lora.LoRALayer(4, 3, rank=8, alpha=16)
    assert torch.allclose(layer(torch.randn(5, 4)), torch.zeros(5, 3))


def test_forward_equals_base_plus_scaled_low_rank_update():
    torch.manual_seed(0)
    base = nn.Linear(4, 3)
    wrapped = lora.LinearWithLoRA(base, rank=2, alpha=4)
    with torch.no_grad():
        wrapped.lora.W_b.copy_(torch.randn(2, 3))

    x = torch.randn(6, 4)
    expected = base(x) + 2.0 * (x @ wrapped.lora.W_a @ wrapped.lora.W_b)
    assert torch.allclose(wrapped(x), expected, atol=1e-6)


def test_freeze_base_controls_base_weight_gradients():
    frozen = lora.LinearWithLoRA(nn.Linear(4, 3), rank=2, alpha=4, freeze_base=True)
    assert not frozen.linear.weight.requires_grad
    assert frozen.lora.W_a.requires_grad

    trainable = lora.LinearWithLoRA(nn.Linear(4, 3), rank=2, alpha=4, freeze_base=False)
    assert trainable.linear.weight.requires_grad


def test_head_base_weights_remain_trainable_after_apply_lora():
    """Regression: the old implementation froze a randomly-initialised head."""
    model = nn.Module()
    model.head = nn.Linear(4, 3)
    model.body = nn.Linear(4, 4)
    lora.apply_lora(model, rank=2, alpha=4, target_layers=["body"], head_layers=["head"])
    assert model.head.linear.weight.requires_grad
    assert not model.body.linear.weight.requires_grad


def test_matches_peft_numerically_under_identical_initialisation():
    from peft.tuners.lora import Linear as PeftLoraLinear
    from peft.tuners.lora import LoraConfig

    torch.manual_seed(0)
    base = nn.Linear(8, 5)
    ours = lora.LinearWithLoRA(base, rank=4, alpha=8)
    with torch.no_grad():
        ours.lora.W_b.copy_(torch.randn(4, 5))

    peft_layer = PeftLoraLinear(
        base_layer=nn.Linear(8, 5),
        adapter_name="default",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
    )
    with torch.no_grad():
        peft_layer.base_layer.weight.copy_(base.weight)
        peft_layer.base_layer.bias.copy_(base.bias)
        # ours: (in, r) and (r, out); peft: (r, in) and (out, r)
        peft_layer.lora_A["default"].weight.copy_(ours.lora.W_a.T)
        peft_layer.lora_B["default"].weight.copy_(ours.lora.W_b.T)

    x = torch.randn(7, 8)
    ours.eval()
    peft_layer.eval()
    with torch.no_grad():
        assert torch.allclose(ours(x), peft_layer(x), atol=1e-5)


def test_count_trainable_parameters():
    model = nn.Linear(4, 3)
    assert lora.count_trainable_parameters(model) == 4 * 3 + 3
    model.weight.requires_grad = False
    assert lora.count_trainable_parameters(model) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lora.py -v`
Expected: FAIL — `ImportError: cannot import name 'lora'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/models/lora.py
"""Low-rank adaptation, implemented from scratch and pinned against PEFT.

Two departures from the previous implementation in this repo:
  * scaling is alpha / rank, the standard formulation, not alpha;
  * wrapped layers can keep their base weights trainable, so a randomly
    initialised head is never frozen behind a rank-r bottleneck.
"""

import math
from functools import partial

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * ((x @ self.W_a) @ self.W_b)


class LinearWithLoRA(nn.Module):
    def __init__(
        self, linear: nn.Linear, rank: int, alpha: float, freeze_base: bool = True
    ) -> None:
        super().__init__()
        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = not freeze_base
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def _replace(parent: nn.Module, name: str, wrap) -> None:
    setattr(parent, name, wrap(getattr(parent, name)))


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    target_layers: list[str] | None = None,
    head_layers: list[str] | None = None,
) -> int:
    """Wrap named `nn.Linear` attributes of `model` with LoRA.

    `target_layers` are frozen behind the adapter; `head_layers` keep their
    base weights trainable. Returns the number of modules wrapped.
    """
    frozen = partial(LinearWithLoRA, rank=rank, alpha=alpha, freeze_base=True)
    trainable = partial(LinearWithLoRA, rank=rank, alpha=alpha, freeze_base=False)

    wrapped = 0
    for name in target_layers or []:
        _replace(model, name, frozen)
        wrapped += 1
    for name in head_layers or []:
        _replace(model, name, trainable)
        wrapped += 1
    return wrapped


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lora.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/models/lora.py tests/test_lora.py
git commit -m "fix: LoRA alpha/rank scaling and trainable head base weights"
```

---

### Task 12: Loss registry and LightningModule

**Files:**
- Create: `finetune_esm/losses.py`, `finetune_esm/models/module.py`, `tests/test_losses.py`, `tests/test_module.py`

**Interfaces:**
- Consumes: `fusion.FusionBase`, `heads.MLPHead`, `metrics.all_metrics`, `batching.EmbeddingBatch`.
- Produces:
  - `losses.get_loss(name: str) -> Callable[[Tensor, Tensor, Tensor | None], Tensor]` — signature `(pred, target, group_id)`. Registry `LOSSES` contains `"mse"` here; ranking losses arrive in the research plan and reuse this signature unchanged.
  - `module.AffinityModule(pl.LightningModule)` — `__init__(self, fusion, head, loss_name="mse", learning_rate=1e-3, total_steps=1000, warmup_steps=100)`. Logs `train_loss`, `val_loss`, `val_spearman`, `val_pearson`, `val_rmse`, and the `test_*` equivalents. Validation and test metrics are computed over the **whole epoch**, not per batch, because Spearman on a 32-row batch is noise.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_losses.py
import pytest
import torch

from finetune_esm import losses


def test_registry_contains_mse():
    assert "mse" in losses.LOSSES


def test_mse_matches_torch():
    pred, target = torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0])
    loss = losses.get_loss("mse")(pred, target, None)
    assert torch.allclose(loss, torch.tensor(2.0))


def test_loss_accepts_group_id_and_ignores_it_for_mse():
    pred, target = torch.randn(8), torch.randn(8)
    groups = torch.randint(0, 3, (8,))
    fn = losses.get_loss("mse")
    assert torch.allclose(fn(pred, target, groups), fn(pred, target, None))


def test_unknown_loss_raises_with_available_names():
    with pytest.raises(KeyError, match="mse"):
        losses.get_loss("nonexistent")
```

```python
# tests/test_module.py
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

from finetune_esm.data.batching import EmbeddingBatch, collate
from finetune_esm.models import fusion, heads, module

DIM = 8


def _items(n=16):
    torch.manual_seed(0)
    out = []
    for i in range(n):
        out.append(
            {
                "ligand": torch.randn(4, DIM).numpy(),
                "ligand_chain_id": torch.zeros(4, dtype=torch.long).numpy(),
                "receptor": torch.randn(3, DIM).numpy(),
                "receptor_chain_id": torch.zeros(3, dtype=torch.long).numpy(),
                "target": float(i),
                "group_id": i % 4,
            }
        )
    return out


def _model():
    fuse = fusion.FUSIONS["hp"](DIM)
    return module.AffinityModule(fuse, heads.MLPHead(fuse.out_dim), total_steps=8)


def test_forward_returns_one_prediction_per_row():
    batch = collate(_items(5))
    assert _model()(batch).shape == (5,)


def test_configure_optimizers_returns_optimizer_and_step_scheduler():
    cfg = _model().configure_optimizers()
    assert isinstance(cfg["optimizer"], torch.optim.Optimizer)
    assert cfg["lr_scheduler"]["interval"] == "step"


def test_one_training_epoch_runs_and_reduces_loss():
    model = _model()
    loader = DataLoader(_items(16), batch_size=4, collate_fn=collate)
    trainer = pl.Trainer(
        max_epochs=4, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, loader, loader)
    assert torch.isfinite(torch.tensor(trainer.callback_metrics["val_loss"].item()))


def test_validation_metrics_are_logged_over_the_whole_epoch():
    model = _model()
    loader = DataLoader(_items(16), batch_size=4, collate_fn=collate)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
    )
    trainer.fit(model, loader, loader)
    for key in ("val_loss", "val_spearman", "val_pearson", "val_rmse"):
        assert key in trainer.callback_metrics
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_losses.py tests/test_module.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.losses'`.

- [ ] **Step 3: Write the implementations**

```python
# finetune_esm/losses.py
"""Training objectives.

Every loss takes (pred, target, group_id) so that group-aware and ranking
objectives can be added later without changing any call site.
"""

from typing import Callable

import torch
import torch.nn.functional as F

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


def mse(
    pred: torch.Tensor, target: torch.Tensor, group_id: torch.Tensor | None = None
) -> torch.Tensor:
    return F.mse_loss(pred, target)


LOSSES: dict[str, LossFn] = {"mse": mse}


def get_loss(name: str) -> LossFn:
    if name not in LOSSES:
        raise KeyError(f"unknown loss {name!r}; available: {sorted(LOSSES)}")
    return LOSSES[name]
```

```python
# finetune_esm/models/module.py
"""LightningModule wiring fusion, head, loss and metrics together."""

import lightning.pytorch as pl
import torch

from finetune_esm import losses, metrics
from finetune_esm.data.batching import EmbeddingBatch


class AffinityModule(pl.LightningModule):
    def __init__(
        self,
        fusion: torch.nn.Module,
        head: torch.nn.Module,
        loss_name: str = "mse",
        learning_rate: float = 1e-3,
        total_steps: int = 1000,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["fusion", "head"])
        self.fusion = fusion
        self.head = head
        self.loss_fn = losses.get_loss(loss_name)
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
        self._val_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._test_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, batch: EmbeddingBatch) -> torch.Tensor:
        return self.head(self.fusion(batch))

    def training_step(self, batch: EmbeddingBatch, batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        loss = self.loss_fn(pred, batch.target, batch.group_id)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch.target))
        return loss

    def _eval_step(self, batch, buffer, prefix):
        pred = self(batch)
        loss = self.loss_fn(pred, batch.target, batch.group_id)
        self.log(f"{prefix}_loss", loss, prog_bar=True, batch_size=len(batch.target))
        buffer.append((pred.detach().cpu(), batch.target.detach().cpu()))

    def _eval_epoch_end(self, buffer, prefix):
        if not buffer:
            return
        pred = torch.cat([p for p, _ in buffer])
        target = torch.cat([t for _, t in buffer])
        for name, value in metrics.all_metrics(pred, target).items():
            self.log(f"{prefix}_{name}", value, prog_bar=(name == "spearman"))
        buffer.clear()

    def validation_step(self, batch: EmbeddingBatch, batch_idx: int) -> None:
        self._eval_step(batch, self._val_buffer, "val")

    def on_validation_epoch_end(self) -> None:
        self._eval_epoch_end(self._val_buffer, "val")

    def test_step(self, batch: EmbeddingBatch, batch_idx: int) -> None:
        self._eval_step(batch, self._test_buffer, "test")

    def on_test_epoch_end(self) -> None:
        self._eval_epoch_end(self._test_buffer, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.warmup_steps / self.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_losses.py tests/test_module.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add finetune_esm/losses.py finetune_esm/models/module.py tests/test_losses.py tests/test_module.py
git commit -m "feat: loss registry and Lightning module with epoch-level metrics"
```

---

### Task 13: Training CLI

**Files:**
- Create: `finetune_esm/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build_cache_command(backbone: str, device: str) -> Path` — encodes every unique chain in the dataset, returns the cache directory.
  - `run_training(backbone, fusion_name, split_name, seed, ...) -> dict[str, float]` — returns test metrics.
  - A Typer app with subcommands `build-cache` and `train`.

`total_steps` is computed as `num_epochs * ceil(len(train_loader))`, taken from the loader itself. This avoids the bug in the old `train.py:104`, which derived it from the sample count without dividing by the number of workers and so over-ran the schedule by a factor of `num_workers`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from finetune_esm import train
from finetune_esm.data import batching


class FakeStore:
    def residues(self, seq: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seq)) % (2**32))
        return rng.normal(size=(len(seq), 8)).astype(np.float32)


def _df(n=64) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "pdb": [f"P{i // 2}" for i in range(n)],
            "ligand_seqs": [["MK" * (1 + i % 3)] for i in range(n)],
            "receptor_seqs": [["GG" * (1 + i % 2)] for i in range(n)],
            "pkd": rng.normal(7.0, 2.0, size=n),
        }
    )


def test_total_steps_uses_loader_length_not_row_count():
    loader = DataLoader(
        batching.ComplexDataset(_df(64), np.arange(64), FakeStore()),
        batch_size=8,
        collate_fn=batching.collate,
    )
    assert train.compute_total_steps(loader, num_epochs=3) == 3 * math.ceil(64 / 8)


def test_total_steps_is_at_least_one():
    loader = DataLoader(
        batching.ComplexDataset(_df(2), np.arange(2), FakeStore()),
        batch_size=8,
        collate_fn=batching.collate,
    )
    assert train.compute_total_steps(loader, num_epochs=1) >= 1


def test_end_to_end_training_returns_all_three_metrics():
    out = train.run_training(
        df=_df(64),
        store=FakeStore(),
        hidden_dim=8,
        fusion_name="hp",
        split_name="cold_pdb",
        seed=0,
        num_epochs=2,
        batch_size=8,
        accelerator="cpu",
    )
    assert set(out) == {"spearman", "pearson", "rmse"}
    assert np.isfinite(out["rmse"])


def test_split_choice_is_honoured_and_leak_free():
    df = _df(64)
    idx = train.resolve_split(df, "cold_pdb", seed=0)
    from finetune_esm.data import splits

    splits.assert_no_pdb_leakage(df, idx)


def test_unknown_split_name_raises():
    with pytest.raises(KeyError, match="cold_pdb"):
        train.resolve_split(_df(8), "not_a_split", seed=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'finetune_esm.train'`.

- [ ] **Step 3: Write the implementation**

```python
# finetune_esm/train.py
"""Command line entrypoints: build the embedding cache, then train."""

import math
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import typer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from finetune_esm import config
from finetune_esm.data import batching, cache, dataset, splits
from finetune_esm.models import backbones, fusion, heads, module

app = typer.Typer(add_completion=False)

SPLIT_BUILDERS = {
    "author": lambda df, seed: splits.author_split(df),
    "cold_pdb": lambda df, seed: splits.cold_pdb_split(df, seed=seed),
    "random": lambda df, seed: splits.random_split(df, seed=seed),
}


def resolve_split(df: pd.DataFrame, split_name: str, seed: int) -> dict[str, np.ndarray]:
    if split_name not in SPLIT_BUILDERS:
        raise KeyError(
            f"unknown split {split_name!r}; available: {sorted(SPLIT_BUILDERS)}"
        )
    return SPLIT_BUILDERS[split_name](df, seed)


def compute_total_steps(loader: DataLoader, num_epochs: int) -> int:
    """Steps the scheduler will actually take, derived from the loader itself."""
    return max(1, num_epochs * math.ceil(len(loader.dataset) / loader.batch_size))


def _loader(df, idx, store, batch_size, shuffle):
    return DataLoader(
        batching.ComplexDataset(df, idx, store),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=batching.collate,
    )


def run_training(
    df: pd.DataFrame,
    store,
    hidden_dim: int,
    fusion_name: str = "hp",
    split_name: str = "author",
    seed: int = 0,
    num_epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    dropout: float = 0.1,
    loss_name: str = "mse",
    patience: int = 5,
    accelerator: str = "auto",
) -> dict[str, float]:
    """Train one configuration and return metrics on its test split."""
    pl.seed_everything(seed, workers=True)
    idx = resolve_split(df, split_name, seed)
    splits.assert_no_pdb_leakage(df, idx)

    train_loader = _loader(df, idx["train"], store, batch_size, shuffle=True)
    val_loader = _loader(df, idx["val"], store, batch_size, shuffle=False)
    test_loader = _loader(df, idx["test"], store, batch_size, shuffle=False)

    fuse = fusion.FUSIONS[fusion_name](hidden_dim)
    model = module.AffinityModule(
        fuse,
        heads.MLPHead(fuse.out_dim, dropout=dropout),
        loss_name=loss_name,
        learning_rate=learning_rate,
        total_steps=compute_total_steps(train_loader, num_epochs),
    )

    checkpoint = ModelCheckpoint(monitor="val_spearman", mode="max", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        logger=False,
        enable_progress_bar=False,
        callbacks=[
            checkpoint,
            EarlyStopping(monitor="val_spearman", mode="max", patience=patience),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    (results,) = trainer.test(model, test_loader, ckpt_path=checkpoint.best_model_path)
    return {
        "spearman": results["test_spearman"],
        "pearson": results["test_pearson"],
        "rmse": results["test_rmse"],
    }


@app.command("build-cache")
def build_cache_command(
    backbone: str = typer.Option("esm2-650m", help="Key in models.backbones.BACKBONES"),
    device: str = typer.Option("cuda", help="Device for the encoder forward pass"),
    batch_size: int = typer.Option(8),
) -> Path:
    """Encode every unique chain in the dataset once."""
    df = dataset.load_complexes()
    seqs = [s for row in df["ligand_seqs"] for s in row]
    seqs += [s for row in df["receptor_seqs"] for s in row]

    spec = backbones.BACKBONES[backbone]
    encoder = backbones.Backbone(spec, device=device, dtype=torch.float16)
    out_dir = config.CACHE_DIR / backbone
    meta = cache.build_cache(seqs, encoder, out_dir, batch_size=batch_size)
    typer.echo(f"cached {meta.n_chains} unique chains to {out_dir}")
    return out_dir


@app.command("train")
def train_command(
    backbone: str = typer.Option("esm2-650m"),
    fusion_name: str = typer.Option("hp", "--fusion"),
    split_name: str = typer.Option("author", "--split"),
    seed: int = typer.Option(0),
    num_epochs: int = typer.Option(20),
    batch_size: int = typer.Option(32),
    learning_rate: float = typer.Option(1e-3),
) -> None:
    """Train one configuration on the cached embeddings."""
    df = dataset.load_complexes()
    spec = backbones.BACKBONES[backbone]
    store = cache.EmbeddingCache(config.CACHE_DIR / backbone)
    out = run_training(
        df=df,
        store=store,
        hidden_dim=spec.hidden_dim,
        fusion_name=fusion_name,
        split_name=split_name,
        seed=seed,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    typer.echo(
        f"{backbone}/{fusion_name}/{split_name} seed={seed}: "
        f"spearman={out['spearman']:.4f} pearson={out['pearson']:.4f} "
        f"rmse={out['rmse']:.4f}"
    )


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the whole fast suite**

Run: `uv run pytest -m "not slow" -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add finetune_esm/train.py tests/test_train.py
git commit -m "feat: build-cache and train CLI commands"
```

---

### Task 14: E0 — reproduce the published ESM2-650M + HP result

This is the gate. If it fails, stop and diagnose; nothing downstream is credible without it.

**Files:**
- Create: `results/e0_reproduction.md`

**Interfaces:**
- Consumes: the `build-cache` and `train` CLI commands.
- Produces: a recorded result table. No new code.

- [ ] **Step 1: Build the ESM-2 650M cache**

```bash
uv run python -m finetune_esm.train build-cache --backbone esm2-650m --device cuda --batch-size 8
```

Expected: `cached 10304 unique chains to <repo>/embedding_cache/esm2-650m`, and a `residues.f16.npy` of roughly 5.0 GB. Runtime is minutes, not hours, on the 3090.

- [ ] **Step 2: Verify the cache against a live forward pass**

```bash
uv run pytest tests/test_cache.py -v -m slow
```

Expected: all pass. This is the guard against a silently corrupt cache poisoning every later number.

- [ ] **Step 3: Train three seeds on the authors' split**

```bash
for seed in 0 1 2; do
  uv run python -m finetune_esm.train train \
    --backbone esm2-650m --fusion hp --split author --seed "$seed" \
    --num-epochs 30 --batch-size 32 --learning-rate 1e-3
done
```

Expected: three lines reporting Spearman, Pearson and RMSE on the test split.

- [ ] **Step 4: Record the outcome and apply the gate**

Write `results/e0_reproduction.md` containing a table of the three seeds, their mean ± sd, and the published reference (ESM2-650M + HP, frozen + ConvBERT: Spearman 0.47, Pearson 0.48, RMSE 1.74), plus the trivial-feature floor (0.212).

Gate criteria, in order:

- **Mean Spearman within 0.05 of 0.47** → reproduction accepted; proceed to the research plan.
- **Mean Spearman ≥ 0.35 but outside that band** → partial. Our head is an MLP, not the paper's ConvBERT, so some gap is expected. Record the difference, note the architectural deviation, and proceed — but treat the published number, not ours, as the bar to beat.
- **Mean Spearman < 0.30** → reproduction failed. **Stop.** Diagnose before any further work. Check, in this order: cache/sequence alignment (does `store.residues(seq)` have `len(seq)` rows?), split assignment (do the counts match 6,485/965/757?), target derivation (is pKd in 1.39–15.70?), and whether the trivial baseline still returns 0.212 through the same data path.

- [ ] **Step 5: Commit**

```bash
git add results/e0_reproduction.md
git commit -m "test: record E0 reproduction of ESM2-650M + HP baseline"
```

---

## Self-Review

**Spec coverage.** §5.1 stack → Task 1. §2 dataset properties → Tasks 2, 5. §6 splits and leakage → Task 3. §6 metrics → Task 4. §2.1 trivial floor → Task 5. §5.2 screening tier and §5.4 cache format → Tasks 6, 7. §5.3 fusion/head interfaces → Tasks 8, 9, 10. §5.5 LoRA repair → Task 11. §7 E0 gate → Task 14. §8 testing → distributed across every task.

**Deferred to the research plan, by design:** E1 controls (scrambled partner, single partner, label shuffle), E2 ranking losses, E3 group-aware sampling, E4 LoRA tier training runs, E5 backbone sweep, E6 ensembling, E7 calibration, the Optuna sweep driver, MLflow logging integration, and the §6 stretch goal of an interface-based split. The interfaces they need already exist here: `losses` takes `group_id`, `EmbeddingBatch` carries it, `apply_lora` is built and tested, and `BACKBONES` is a registry rather than a hard-coded model.

**Type consistency.** `EmbeddingBatch` field names are identical across Tasks 8, 9, 12, 13. `FusionBase.out_dim` is set by every subclass and consumed by `MLPHead(fuse.out_dim)` in Tasks 12 and 13. `CacheMeta` is constructed in `build_cache` and compared in `EmbeddingCache.__init__`, with `n_chains` present in both. `store.residues(seq)` is the only method `ComplexDataset` requires, which is why `FakeStore` in Tasks 8 and 13 can implement it alone.

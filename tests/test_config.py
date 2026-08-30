from pathlib import Path

from finetune_esm import config


def test_paths_are_absolute_and_under_root():
    assert config.ROOT_DIR.is_absolute()
    for p in (config.CACHE_DIR, config.RESULTS_DIR):
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

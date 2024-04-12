import mlflow
import logging
from pathlib import Path


# Directories
ROOT_DIR = Path(__file__).parent.parent.resolve()
LOGS_DIR = ROOT_DIR / "logs"
STORAGE_DIR = ROOT_DIR / "finetune_results"

# Config MLflow
MODEL_REGISTRY = STORAGE_DIR / "mlflow"
MLFLOW_TRACKING_URI = "file://" + str(MODEL_REGISTRY)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


logger = logging.getLogger("mlflow")
# Set log level to debugging
logger.setLevel(logging.DEBUG)

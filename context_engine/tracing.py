"""MLflow tracing for Context Engine.

Thin wrapper — gracefully degrades to no-ops if mlflow is not installed.
Uses mlflow.anthropic.autolog() to auto-trace all Claude API calls.
Shares tracking DB with Gaia and Iris at ~/.gaia/mlflow/mlflow.db.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.anthropic
    HAS_MLFLOW = True
except ImportError:
    mlflow = None
    HAS_MLFLOW = False

_initialized = False

DEFAULT_TRACKING_DIR = Path.home() / ".gaia" / "mlflow"


def init_tracing(experiment_name: str = "context-engine") -> None:
    global _initialized
    if not HAS_MLFLOW or _initialized:
        return

    tracking_dir = os.environ.get("GAIA_MLFLOW_URI", str(DEFAULT_TRACKING_DIR))
    Path(tracking_dir).mkdir(parents=True, exist_ok=True)
    db_path = Path(tracking_dir) / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(experiment_name)
    mlflow.anthropic.autolog()
    _initialized = True
    logger.info("MLflow tracing initialized: %s (autolog enabled)", experiment_name)

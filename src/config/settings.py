"""
settings.py — Centralised project paths and configuration.

All paths are derived from PROJ_ROOT so the project is location-independent.
Import from this module (or from the config package) anywhere else in the codebase.

Usage
-----
    from ecommerce_customer_value_prediction.config import PROJ_ROOT, RAW_DATA_DIR
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load optional .env from the project root (never required, never committed)
load_dotenv()

# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

# src/config/settings.py
#   → parents[0] = src/config/
#   → parents[1] = src/
#   → parents[2] = project root
PROJ_ROOT: Path = Path(__file__).resolve().parents[2]
logger.info(f"PROJ_ROOT: {PROJ_ROOT}")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

# ---------------------------------------------------------------------------
# Artifact directories
# ---------------------------------------------------------------------------

MODELS_DIR: Path = PROJ_ROOT / "models"

REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_DIR: Path = REPORTS_DIR / "metrics"

# ---------------------------------------------------------------------------
# Loguru → tqdm compatibility
# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except (ModuleNotFoundError, ValueError):
    pass

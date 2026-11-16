"""Paths, constants and config loading shared by every notebook."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "dataset" / "archive" / "UCI_Credit_Card.csv"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "data" / "features"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = OUTPUTS / "reports"
QUERIES = ROOT / "queries"
CONFIG_DIR = ROOT / "config"

for _d in (PROCESSED, FEATURES, FIGURES, TABLES, MODELS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.20
TARGET = "default_next_month"

# Chronological month index 1..6 (Apr..Sep 2005). Raw column suffix k maps to month 7-k,
# and PAY_0 is the September status (suffix 1 in the naming used by BILL/PAY_AMT).
MONTHS = [1, 2, 3, 4, 5, 6]
MONTH_LABELS = {1: "2005-04", 2: "2005-05", 3: "2005-06", 4: "2005-07", 5: "2005-08", 6: "2005-09"}
MONTH_NAMES = {1: "Apr", 2: "May", 3: "Jun", 4: "Jul", 5: "Aug", 6: "Sep"}

# Behavioural thresholds, fixed from the Phase 3 train-split validation tables.
HIGH_UTIL = 0.90
WEAK_COVERAGE = 0.10
FULL_COVERAGE = 0.99
INACTIVE_UTIL = 0.02
UTIL_RISE = 0.05

DEMOGRAPHICS = ["sex", "education", "marriage", "age"]


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)

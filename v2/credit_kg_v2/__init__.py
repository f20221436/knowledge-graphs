"""v2 extensions: expected extra loss, what-would-change explanations, bank-style scorecard.

v2 imports the v1 package (src/credit_kg) read-only and never writes into v1 folders.
"""
import sys
from pathlib import Path

import yaml

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = V2 / "outputs"
OUT.mkdir(exist_ok=True)


def load_config() -> dict:
    with open(V2 / "config_v2.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

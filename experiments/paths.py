"""Repository paths shared by the experiments and the figure script."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

AUDITS_PRIMARY = RESULTS / "primary_audit"
AUDITS_INITIAL = RESULTS / "initial_runs"
OPTIMIZERS_RAM = RESULTS / "ram"
OPTIMIZERS_COM = RESULTS / "com"

"""Shared helpers for the measured audits."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.paths import DATA as DATA_ROOT
from experiments.paths import AUDITS_INITIAL as RESULTS_ROOT
from experiments.paths import AUDITS_PRIMARY as EXTERNAL_OUTPUT
from experiments.paths import ROOT


DATASETS: dict[str, dict[str, int]] = {
    "tf_bind_8": {"alphabet_size": 4, "train": 4096, "validation": 2048},
    "trpb": {"alphabet_size": 20, "train": 4960, "validation": 1240},
    "gb1": {"alphabet_size": 20, "train": 4960, "validation": 620},
}
MODELS = ("pairwise_ridge", "extra_trees", "ranking", "com")
SEEDS = (20270813, 20270814, 20270815, *range(20270901, 20270913))
BUDGETS = (8, 16, 32, 64, 128, 256)
TOLERANCE = 1e-12


def stable_seed(*values: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, values)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(DATA_ROOT / name / "candidates.parquet")
    token_columns = sorted(
        (column for column in frame if column.startswith("token_")),
        key=lambda column: int(column.removeprefix("token_")),
    )
    tokens = frame[token_columns].to_numpy(np.int8)
    reference = frame.f.to_numpy(np.float64)
    candidate_ids = frame.candidate_id.to_numpy(np.int64)
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise AssertionError(f"{name}: candidate IDs are not unique")
    return frame, tokens, reference, candidate_ids


def split_indices(name: str, candidate_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    specification = DATASETS[name]
    order = np.random.default_rng(seed).permutation(len(candidate_ids))
    train_stop = specification["train"]
    validation_stop = train_stop + specification["validation"]
    return {
        "train": np.sort(order[:train_stop]).astype(np.int64),
        "validation": np.sort(order[train_stop:validation_stop]).astype(np.int64),
        "selection": np.sort(order[validation_stop:]).astype(np.int64),
    }


def save_and_validate_split(
    directory: Path,
    name: str,
    seed: int,
    candidate_ids: np.ndarray,
    split: dict[str, np.ndarray],
) -> dict[str, object]:
    split_directory = directory / "raw" / "splits" / name
    split_directory.mkdir(parents=True, exist_ok=True)
    path = split_directory / f"seed_{seed}.npz"
    ids = {key: candidate_ids[split[key]].astype(np.int64) for key in ("train", "validation", "selection")}
    np.savez_compressed(path, **ids)
    persisted = np.load(path, allow_pickle=False)
    train = persisted["train"]
    validation = persisted["validation"]
    selection = persisted["selection"]
    overlaps = {
        "train_validation_overlap": int(np.intersect1d(train, validation, assume_unique=True).size),
        "train_selection_overlap": int(np.intersect1d(train, selection, assume_unique=True).size),
        "validation_selection_overlap": int(np.intersect1d(validation, selection, assume_unique=True).size),
    }
    total_unique = np.unique(np.concatenate((train, validation, selection))).size
    passed = not any(overlaps.values()) and total_unique == len(candidate_ids)
    if not passed:
        raise AssertionError(f"{name}/{seed}: independently persisted split has leakage")
    return {
        "dataset": name,
        "seed": seed,
        "path": str(path.relative_to(ROOT)),
        "train_size": len(train),
        "validation_size": len(validation),
        "selection_size": len(selection),
        **overlaps,
        "all_ids_once": bool(total_unique == len(candidate_ids)),
        "passed": passed,
    }


def prior_prediction_path(name: str, seed: int, model: str) -> Path:
    return RESULTS_ROOT / name / "predictions" / f"seed_{seed}_{model}.npz"


def prediction_path(name: str, seed: int, model: str) -> Path:
    return EXTERNAL_OUTPUT / "raw" / "predictions" / name / f"seed_{seed}_{model}.npz"


def resolve_prediction_path(name: str, seed: int, model: str) -> Path:
    prior = prior_prediction_path(name, seed, model)
    if prior.exists():
        return prior
    return prediction_path(name, seed, model)


def load_prediction(name: str, seed: int, model: str) -> tuple[np.ndarray, float, dict[str, object], Path]:
    path = resolve_prediction_path(name, seed, model)
    saved = np.load(path, allow_pickle=False)
    return (
        saved["prediction"].astype(np.float64),
        float(saved["validation_rmse"]),
        json.loads(str(saved["details"].item())),
        path,
    )


def ranked_order(scores: np.ndarray, pool: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    return pool[np.lexsort((candidate_ids[pool], -scores[pool]))]


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    if not len(values):
        return {f"{prefix}_{name}": np.nan for name in ("min", "q25", "median", "q75", "max")}
    result = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0))
    return dict(zip((f"{prefix}_{name}" for name in ("min", "q25", "median", "q75", "max")), map(float, result)))


def bootstrap_interval(values: Iterable[float], statistic: str, seed: int) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(len(array), size=(10_000, len(array)))]
    sampled = np.mean(draws, axis=1) if statistic == "mean" else np.median(draws, axis=1)
    low, high = np.quantile(sampled, (0.025, 0.975))
    return float(low), float(high)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z**2 / total
    center = (probability + z**2 / (2 * total)) / denominator
    radius = z * np.sqrt(probability * (1 - probability) / total + z**2 / (4 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "No applicable rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not np.isfinite(value) else f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def environment_lines(command: str, elapsed_seconds: float, status: str = "completed") -> list[str]:
    import scipy
    import sklearn
    import torch

    return [
        f"finished_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"torch={torch.__version__}",
        f"elapsed_seconds={elapsed_seconds:.6f}",
        f"status={status}",
    ]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

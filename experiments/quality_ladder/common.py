"""Data loading and helpers for the quality ladder."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/quality_ladder"
DATA = ROOT / "data"
CALIBRATION = OUTPUT / "oracle_quality_calibration.csv"
LADDER_ROWS = OUTPUT / "oracle_ladder.csv"

LANDSCAPES = ("av_gfp", "gb1", "tf_bind_8", "trpb")
BUDGETS = (8, 16, 32, 64)
DELTAS = (0.0, 0.05, 0.10, 0.25, 0.50)
TARGETS = (0.20, 0.40, 0.60, 0.80, 0.95)
ORACLE_REPETITIONS = 30
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Partition:
    landscape: str
    split: int
    candidate_ids: np.ndarray
    reference: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    selection: np.ndarray


def stable_seed(*values: object) -> int:
    payload = "|".join(map(str, values)).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def _ids_to_indices(candidate_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    order = np.argsort(candidate_ids, kind="stable")
    locations = np.searchsorted(candidate_ids[order], requested)
    if np.any(locations >= len(candidate_ids)):
        raise AssertionError("split contains an unknown candidate ID")
    indices = order[locations]
    if not np.array_equal(candidate_ids[indices], requested):
        raise AssertionError("split contains an unknown candidate ID")
    return indices.astype(np.int64)


def load_partition(landscape: str, split: int) -> Partition:
    table = pd.read_parquet(DATA / landscape / "candidates.parquet")
    candidate_ids = table.candidate_id.to_numpy(np.int64)
    reference = table.f.to_numpy(np.float64)
    if landscape == "av_gfp":
        split_frame = pd.read_parquet(
            ROOT / "results/initial_runs/av_gfp/splits.parquet"
        )
        split_frame = split_frame[split_frame.seed.eq(split)]
        parts = {
            name: split_frame.loc[
                split_frame.split.eq(name), "candidate_index"
            ].to_numpy(np.int64)
            for name in ("train", "validation", "selection")
        }
    else:
        split_path = (
            ROOT
            / "results/primary_audit/raw/splits"
            / landscape
            / f"seed_{split}.npz"
        )
        with np.load(split_path, allow_pickle=False) as saved:
            if "selection_indices" in saved.files:
                parts = {
                    name: saved[f"{name}_indices"].astype(np.int64)
                    for name in ("train", "validation", "selection")
                }
                for name, indices in parts.items():
                    id_key = f"{name}_candidate_ids"
                    if id_key in saved.files and not np.array_equal(
                        candidate_ids[indices], saved[id_key]
                    ):
                        raise AssertionError(f"{landscape}/{split}: {name} ID mismatch")
            else:
                parts = {
                    name: _ids_to_indices(candidate_ids, saved[name].astype(np.int64))
                    for name in ("train", "validation", "selection")
                }
    combined = np.concatenate([parts["train"], parts["validation"], parts["selection"]])
    if len(combined) != len(reference) or len(np.unique(combined)) != len(reference):
        raise AssertionError(f"{landscape}/{split}: split is not an exact partition")
    return Partition(
        landscape=landscape,
        split=int(split),
        candidate_ids=candidate_ids,
        reference=reference,
        train=parts["train"],
        validation=parts["validation"],
        selection=parts["selection"],
    )


def split_path(landscape: str, split: int) -> Path:
    if landscape == "av_gfp":
        return ROOT / "results/initial_runs/av_gfp/splits.parquet"
    return (
        ROOT
        / "results/primary_audit/raw/splits"
        / landscape
        / f"seed_{split}.npz"
    )


def top_tail_indices(partition: Partition) -> np.ndarray:
    validation = partition.validation
    count = max(2, int(np.ceil(0.01 * len(validation))))
    return validation[
        np.lexsort(
            (
                partition.candidate_ids[validation],
                -partition.reference[validation],
            )
        )[:count]
    ]


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = stats.spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else np.nan


def selected_index(score: np.ndarray, partition: Partition) -> int:
    pool = partition.selection
    pool_score = score[pool]
    maximum = np.max(pool_score)
    tied = pool[pool_score == maximum]
    return int(tied[np.argmin(partition.candidate_ids[tied])])


def improvement_mask(values: np.ndarray, selected_value: float, delta: float) -> np.ndarray:
    if delta == 0.0:
        return values > selected_value + TOLERANCE
    return values >= selected_value + delta


def first_witness_rank(
    score: np.ndarray,
    candidate_ids: np.ndarray,
    eligible: np.ndarray,
    witness_mask: np.ndarray,
) -> int:
    if not witness_mask.any():
        return -1
    witness_indices = eligible[witness_mask]
    witness_scores = score[witness_indices]
    best_score = np.max(witness_scores)
    tied_witnesses = witness_indices[witness_scores == best_score]
    first_id = int(np.min(candidate_ids[tied_witnesses]))
    eligible_scores = score[eligible]
    better = np.sum(eligible_scores > best_score)
    earlier_ties = np.sum(
        (eligible_scores == best_score) & (candidate_ids[eligible] < first_id)
    )
    return int(1 + better + earlier_ties)


def standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values, ddof=0))
    if scale <= 0:
        raise AssertionError("cannot standardize constant values")
    return (values - float(np.mean(values))) / scale


def selection_tail(partition: Partition) -> np.ndarray:
    pool = partition.selection
    count = max(2, int(np.ceil(0.01 * len(pool))))
    return pool[
        np.lexsort(
            (partition.candidate_ids[pool], -partition.reference[pool])
        )[:count]
    ]


def ndcg_at_one_percent(score: np.ndarray, partition: Partition) -> float:
    pool = partition.selection
    values = partition.reference[pool]
    relevance = values - float(np.min(values))
    count = max(2, int(np.ceil(0.01 * len(pool))))
    discounts = np.log2(np.arange(2, count + 2, dtype=float))
    predicted = np.lexsort((partition.candidate_ids[pool], -score[pool]))[:count]
    ideal = np.lexsort((partition.candidate_ids[pool], -values))[:count]
    denominator = float(np.sum(relevance[ideal] / discounts))
    if denominator <= 0:
        return np.nan
    return float(np.sum(relevance[predicted] / discounts) / denominator)

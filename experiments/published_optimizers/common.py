"""Shared utilities for the published-optimizer runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.paths import DATA as DATA_ROOT

SEEDS = (20270813, 20270814, 20270815)
BUDGETS = (8, 16, 32, 64, 128, 256)
DATASETS: dict[str, dict[str, int]] = {
    "tf_bind_8": {"alphabet_size": 4, "train_size": 4096, "validation_size": 2048},
    "trpb": {"alphabet_size": 20, "train_size": 4960, "validation_size": 1240},
    "gb1": {"alphabet_size": 20, "train_size": 4960, "validation_size": 620},
}
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
    tokens = frame[token_columns].to_numpy(np.int32)
    reference = frame.f.to_numpy(np.float64)
    candidate_ids = frame.candidate_id.to_numpy(np.int64)
    if len(np.unique(candidate_ids)) != len(candidate_ids):
        raise AssertionError(f"{name}: duplicate candidate identifiers")
    if not np.all(np.isfinite(reference)):
        raise AssertionError(f"{name}: non-finite trusted values")
    return frame, tokens, reference, candidate_ids


def make_split(name: str, candidate_ids: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    spec = DATASETS[name]
    order = np.random.default_rng(seed).permutation(len(candidate_ids))
    train_end = spec["train_size"]
    validation_end = train_end + spec["validation_size"]
    split = {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:validation_end]),
        "selection": np.sort(order[validation_end:]),
    }
    id_sets = {key: set(candidate_ids[value].tolist()) for key, value in split.items()}
    overlaps = {
        "train_validation": len(id_sets["train"] & id_sets["validation"]),
        "train_selection": len(id_sets["train"] & id_sets["selection"]),
        "validation_selection": len(id_sets["validation"] & id_sets["selection"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"{name} seed {seed}: split leakage {overlaps}")
    combined = np.concatenate(tuple(split.values()))
    if len(combined) != len(candidate_ids) or len(np.unique(combined)) != len(candidate_ids):
        raise AssertionError(f"{name} seed {seed}: splits are not an exact partition")
    return split


def design_bench_logits(tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    """Exact DiscreteDataset.to_logits transform at author default interpolation 0.6."""
    one_hot = np.eye(alphabet_size, dtype=np.float32)[tokens]
    soft = 0.6 * one_hot + 0.4 / float(alphabet_size)
    log_soft = np.log(soft)
    return (log_soft[:, :, 1:] - log_soft[:, :, :1]).astype(np.float32)


def logits_to_integers(logits: np.ndarray) -> np.ndarray:
    """Exact Design-Bench inverse: pad class-zero logit and take argmax."""
    padding = [(0, 0)] * (logits.ndim - 1) + [(1, 0)]
    return np.argmax(np.pad(logits, padding), axis=-1).astype(np.int32)


def normalize_from_train(
    values: np.ndarray, train: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train].mean(axis=0, keepdims=True)
    scale = values[train].std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return ((values - mean) / scale).astype(np.float32), mean.astype(np.float32), scale.astype(np.float32)


def deterministic_order(
    scores: np.ndarray, indices: np.ndarray, candidate_ids: np.ndarray
) -> np.ndarray:
    return indices[np.lexsort((candidate_ids[indices], -scores[indices]))]


def proposal_indices(
    proposal_tokens: np.ndarray,
    tokens: np.ndarray,
    selection: np.ndarray,
) -> tuple[np.ndarray, int]:
    selection_lookup = {tokens[index].tobytes(): int(index) for index in selection}
    eligible = [selection_lookup[row.tobytes()] for row in proposal_tokens if row.tobytes() in selection_lookup]
    unique = np.asarray(list(dict.fromkeys(eligible)), dtype=np.int64)
    return unique, len(proposal_tokens) - len(eligible)


def proposal_diagnostics(
    proposal_tokens: np.ndarray,
    tokens: np.ndarray,
    split: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
) -> list[dict[str, Any]]:
    lookup = {tokens[index].tobytes(): int(index) for index in range(len(tokens))}
    membership = {
        int(index): name for name in ("train", "validation", "selection") for index in split[name]
    }
    rows = []
    for proposal_number, token_row in enumerate(proposal_tokens):
        index = lookup.get(token_row.tobytes())
        rows.append({
            "proposal_number": proposal_number,
            "tokens": json.dumps(token_row.tolist(), separators=(",", ":")),
            "candidate_index": index,
            "candidate_id": int(candidate_ids[index]) if index is not None else None,
            "split": membership.get(index, "not_in_measured_table") if index is not None else "not_in_measured_table",
        })
    return rows


def audit_predictions(
    *,
    method: str,
    dataset: str,
    seed: int,
    prediction: np.ndarray,
    reference: np.ndarray,
    candidate_ids: np.ndarray,
    selection: np.ndarray,
    delivered: np.ndarray,
    proposal_count: int,
    ineligible_proposal_count: int,
    validation_rmse: float,
    training_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if prediction.shape != reference.shape or not np.all(np.isfinite(prediction)):
        raise AssertionError(f"{method} {dataset} {seed}: invalid predictions")
    proxy_order = deterministic_order(prediction, selection, candidate_ids)
    exact = int(proxy_order[0])
    true_order = deterministic_order(reference, selection, candidate_ids)
    true_optimum = int(true_order[0])
    if delivered.size:
        delivered_order = deterministic_order(prediction, delivered, candidate_ids)
        delivered_nomination: int | None = int(delivered_order[0])
    else:
        delivered_nomination = None
    nominations = {
        "exact_learned_score_optimum": exact,
        "method_delivered": delivered_nomination,
    }
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "method": method,
        "proposal_count": proposal_count,
        "eligible_unique_proposal_count": int(len(delivered)),
        "ineligible_proposal_count": int(ineligible_proposal_count),
        "exact_learned_score_optimum_index": exact,
        "method_delivered_nomination_index": delivered_nomination,
        "true_optimum_index": true_optimum,
    }
    for nomination_type, nomination in nominations.items():
        if nomination is None:
            for budget in BUDGETS:
                rows.append({
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "nomination_type": nomination_type,
                    "budget": budget,
                    "status": "no_eligible_method_proposal",
                    "selection_size": len(selection),
                    "proposal_count": proposal_count,
                    "eligible_unique_proposal_count": len(delivered),
                    "ineligible_proposal_count": ineligible_proposal_count,
                    "training_seconds": training_seconds,
                    "validation_rmse": validation_rmse,
                })
            continue
        probe_order = np.concatenate(([nomination], proxy_order[proxy_order != nomination]))
        regret = max(0.0, float(reference[true_optimum] - reference[nomination]))
        witnesses = np.flatnonzero(reference[probe_order] > reference[nomination] + TOLERANCE)
        witness_rank = int(witnesses[0] + 1) if len(witnesses) else None
        if (regret > TOLERANCE) != (witness_rank is not None):
            raise AssertionError(f"{method} {dataset} {seed}: witness/regret mismatch")
        for budget in BUDGETS:
            trusted = probe_order[: min(budget, len(probe_order))]
            lower = max(0.0, float(reference[trusted].max() - reference[nomination]))
            if lower > regret + 1e-9:
                raise AssertionError(f"{method} {dataset} {seed}: certificate exceeds regret")
            certified = lower > TOLERANCE
            if certified != (witness_rank is not None and witness_rank <= budget):
                raise AssertionError(f"{method} {dataset} {seed}: witness-rank identity failed")
            rows.append({
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "nomination_type": nomination_type,
                "status": "complete",
                "budget": budget,
                "selection_size": len(selection),
                "nomination_index": nomination,
                "nomination_candidate_id": int(candidate_ids[nomination]),
                "true_optimum_index": true_optimum,
                "exact_learned_score_optimum_index": exact,
                "proposal_count": proposal_count,
                "eligible_unique_proposal_count": len(delivered),
                "ineligible_proposal_count": ineligible_proposal_count,
                "g_verifier": regret,
                # This audit treats each reported nomination as x_sel.  With
                # exhaustive trusted reference ranking, its trusted total
                # regret is therefore the same finite quantity.
                "g_total": regret,
                "verifier_failure": regret > TOLERANCE,
                "g_search_proxy": max(0.0, float(prediction[exact] - prediction[nomination])),
                "witness_rank": witness_rank,
                "l_b": lower,
                "l_b_over_g_verifier": lower / regret if regret > TOLERANCE else np.nan,
                "certified": certified,
                "validation_rmse": validation_rmse,
                "training_seconds": training_seconds,
            })
    return rows, diagnostics


def save_split(path: Path, split: dict[str, np.ndarray], candidate_ids: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_indices=split["train"],
        validation_indices=split["validation"],
        selection_indices=split["selection"],
        train_candidate_ids=candidate_ids[split["train"]],
        validation_candidate_ids=candidate_ids[split["validation"]],
        selection_candidate_ids=candidate_ids[split["selection"]],
    )


def environment_record() -> dict[str, Any]:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "command": " ".join(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_results_markdown(path: Path, method: str, frame: pd.DataFrame, extra: str) -> None:
    complete = frame[frame.status == "complete"].copy() if "status" in frame else frame.copy()
    exact = complete[complete.nomination_type == "exact_learned_score_optimum"]
    delivered = complete[complete.nomination_type == "method_delivered"]
    lines = [
        f"# {method} official-code audit",
        "",
        "## Setup",
        "",
        extra,
        "",
        "## Main results",
        "",
    ]
    if complete.empty:
        lines.append("No integrated audit completed; see `sanity_checks.json` and `run.log`.")
    else:
        for name, group in (("Exact learned-score optimum", exact), ("Method-delivered", delivered)):
            if group.empty:
                lines.append(f"- {name}: no eligible completed nominations.")
                continue
            per_run = group.sort_values("budget").groupby(["dataset", "seed"], as_index=False).last()
            lines.append(
                f"- {name}: {len(per_run)} dataset/split runs; mean final-budget trusted regret "
                f"{per_run.g_verifier.mean():.6g}."
            )
    lines.extend(["", "## Key numbers", ""])
    if not complete.empty:
        summary_rows = []
        for keys, group in complete.groupby(["dataset", "nomination_type", "budget"]):
            genuine = group[group.g_verifier > TOLERANCE]
            summary_rows.append({
                "dataset": keys[0],
                "nomination_type": keys[1],
                "budget": keys[2],
                "runs": group.seed.nunique(),
                "verifier_failure_runs": genuine.seed.nunique(),
                "mean_g_verifier": group.g_verifier.mean(),
                "mean_g_total": group.g_total.mean(),
                "certification_recall": genuine.certified.mean() if len(genuine) else np.nan,
                "mean_tightness": genuine.l_b_over_g_verifier.mean() if len(genuine) else np.nan,
            })
        summary = pd.DataFrame(summary_rows)
        columns = list(summary.columns)
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for record in summary.to_dict("records"):
            lines.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    else:
        lines.append("No numerical results.")
    lines.extend([
        "",
        "## Negative or unexpected results",
        "",
        "Native dependency failures and any proposal eligibility failures are retained in `sanity_checks.json`, "
        "`diagnostics.parquet`, and `run.log`.",
        "",
        "## Files",
        "",
        "- `metrics.parquet`: budget-level audit metrics.",
        "- `raw_results.parquet`: identical minimally processed budget-level rows.",
        "- `diagnostics.parquet`: nomination and proposal diagnostics.",
        "- `raw/`: model predictions, proposals, splits, and checkpoints.",
        "- `config.json`: exact run and adaptation configuration.",
        "- `provenance.json`: repository and environment provenance.",
        "- `sanity_checks.json`: native and component sanity outcomes.",
        "- `run.log`: run chronology and failures.",
        "",
    ])
    path.write_text("\n".join(lines))

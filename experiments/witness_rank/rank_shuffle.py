"""Rank-shuffle ablation (Section 5.2)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.special import gammaln

from experiments.witness_rank.common import (
    OUTPUT,
    input_paths,
    instances,
    is_improvement,
    load_instance,
    markdown_table,
    stable_seed,
    write_manifest,
)


BUDGETS = (8, 16, 64)
SEEDS = tuple(range(200))
CONDITIONS = ("global_shuffle", "score_decile_shuffle", "top_1pct_shuffle", "identity")


def first_in_random_group(
    size: int, witness_count: int, rng: np.random.Generator, draws: int
) -> np.ndarray:
    if witness_count == 0:
        return np.full(draws, -1, dtype=np.int64)
    # Invert P(L > k) = C(size-witness_count,k) / C(size,k), where L is
    # the one-indexed first-witness position. Vectorised binary search is
    # much faster than scipy's scalar generic sampler at these pool sizes.
    log_targets = np.log1p(-rng.random(draws))
    low = np.zeros(draws, dtype=np.int64)
    high = np.full(draws, size - witness_count + 1, dtype=np.int64)
    constant = gammaln(size - witness_count + 1) - gammaln(size + 1)
    while np.any(low < high):
        active = low < high
        middle = (low + high) // 2
        valid = active & (middle <= size - witness_count)
        log_survival = np.full(draws, -np.inf)
        k = middle[valid]
        log_survival[valid] = (
            constant
            - gammaln(size - witness_count - k + 1)
            + gammaln(size - k + 1)
        )
        move_right = active & (log_survival > log_targets)
        low = np.where(move_right, middle + 1, low)
        high = np.where(active & ~move_right, middle, high)
    return low


def decile_groups(scores: np.ndarray, candidate_indices: np.ndarray, nomination: int) -> list[np.ndarray]:
    cuts = np.quantile(scores, np.arange(0.1, 1.0, 0.1))
    bins = np.searchsorted(cuts, scores, side="right")
    result = []
    for label in range(9, -1, -1):
        values = candidate_indices[bins == label]
        result.append(values[values != nomination])
    if sum(map(len, result)) != len(candidate_indices) - 1:
        raise AssertionError("score-decile partition did not cover every non-nomination candidate")
    return result


def shuffled_ranks(
    condition: str,
    order: np.ndarray,
    selection: np.ndarray,
    prediction: np.ndarray,
    witness_lookup: np.ndarray,
    groups: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    base_positions = np.flatnonzero(witness_lookup[order])
    if not len(base_positions):
        return np.full(len(SEEDS), -1, dtype=np.int64)
    if condition == "identity":
        return np.full(len(SEEDS), int(base_positions[0] + 1), dtype=np.int64)
    remainder = order[1:]
    if condition == "global_shuffle":
        local = first_in_random_group(
            len(remainder), int(witness_lookup[remainder].sum()), rng, len(SEEDS)
        )
        return np.where(local == -1, -1, local + 1)
    if condition == "score_decile_shuffle":
        offset = 0
        for group in groups:
            witness_count = int(witness_lookup[group].sum())
            if witness_count:
                local = first_in_random_group(len(group), witness_count, rng, len(SEEDS))
                return 1 + offset + local
            offset += len(group)
        return np.full(len(SEEDS), -1, dtype=np.int64)
    if condition == "top_1pct_shuffle":
        top_count = max(2, int(math.ceil(0.01 * len(selection))))
        top = order[1:top_count]
        witness_count = int(witness_lookup[top].sum())
        if witness_count:
            local = first_in_random_group(len(top), witness_count, rng, len(SEEDS))
            return local + 1
        later = np.flatnonzero(witness_lookup[order[top_count:]])
        value = int(top_count + later[0] + 1) if len(later) else -1
        return np.full(len(SEEDS), value, dtype=np.int64)
    raise ValueError(condition)


def compute() -> tuple[pd.DataFrame, dict[str, object]]:
    p1 = pd.read_csv(OUTPUT / "margin_nomination/margin_metrics.csv")
    primary = p1[p1.delta_family.eq("sd") & p1.delta_multiplier.eq(0.05)].set_index("instance_id")
    per_instance: list[dict[str, object]] = []
    for number, item in enumerate(instances(), start=1):
        loaded = load_instance(item)
        selected_value = float(loaded.reference[loaded.order[0]])
        delta = float(primary.loc[item.instance_id].delta)
        zero = np.zeros(len(loaded.reference), dtype=bool)
        zero[loaded.selection] = is_improvement(
            loaded.reference[loaded.selection], selected_value, 0.0
        )
        margin = np.zeros(len(loaded.reference), dtype=bool)
        margin[loaded.selection] = is_improvement(
            loaded.reference[loaded.selection], selected_value, delta
        )
        groups = decile_groups(
            loaded.prediction[loaded.selection], loaded.selection, int(loaded.order[0])
        )
        for condition in CONDITIONS:
            rng_zero = np.random.default_rng(stable_seed("P8", item.instance_id, condition, "zero"))
            rng_delta = np.random.default_rng(stable_seed("P8", item.instance_id, condition, "delta"))
            ranks = shuffled_ranks(condition, loaded.order, loaded.selection, loaded.prediction, zero, groups, rng_zero)
            delta_ranks = shuffled_ranks(condition, loaded.order, loaded.selection, loaded.prediction, margin, groups, rng_delta)
            for seed, rank, rank_delta in zip(SEEDS, ranks, delta_ranks):
                row = {
                    "instance_id": item.instance_id,
                    "landscape": item.landscape,
                    "model": item.model,
                    "split": item.split,
                    "condition": condition,
                    "shuffle_seed": seed,
                    "R": rank,
                    "R_delta": rank_delta,
                    "delta": delta,
                }
                for budget in BUDGETS:
                    row[f"recall_{budget}"] = bool(rank != -1 and rank <= budget)
                    row[f"recall_delta_{budget}"] = bool(rank_delta != -1 and rank_delta <= budget)
                per_instance.append(row)
        print(json.dumps({"run": "P8", "instance": number, "total": 230, "id": item.instance_id}), flush=True)
    raw = pd.DataFrame(per_instance)
    rows = []
    for key, group in raw.groupby(["landscape", "model", "condition", "shuffle_seed"], sort=True):
        finite = group.R[group.R.ne(-1)]
        finite_delta = group.R_delta[group.R_delta.ne(-1)]
        row = {
            "landscape": key[0],
            "model": key[1],
            "condition": key[2],
            "shuffle_seed": key[3],
            "audits": len(group),
            "median_R": float(finite.median()) if len(finite) else np.nan,
            "median_R_delta_given_exists": float(finite_delta.median()) if len(finite_delta) else np.nan,
        }
        for budget in BUDGETS:
            row[f"recall_{budget}"] = float(group[f"recall_{budget}"].mean())
            row[f"recall_delta_{budget}"] = float(group[f"recall_delta_{budget}"].mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    identity = raw[raw.condition.eq("identity") & raw.shuffle_seed.eq(0)].set_index("instance_id")
    expected = primary.R_delta
    identity_mismatches = int((identity.R_delta.reindex(expected.index) != expected).sum())
    p3 = pd.read_csv(OUTPUT / "random_order_null/metrics.csv")
    p3_primary = p3[p3.delta_family.eq("sd") & p3.delta_multiplier.eq(0.05)]
    global_rows = raw[raw.condition.eq("global_shuffle")]
    comparisons = []
    for budget in BUDGETS:
        observed = float(global_rows[f"recall_delta_{budget}"].mean())
        exact = float(p3_primary[f"recall_random_{budget}"].mean())
        trials = len(global_rows)
        se = math.sqrt(max(exact * (1.0 - exact), 1e-12) / trials)
        comparisons.append({"budget": budget, "observed": observed, "exact": exact, "absolute_error": abs(observed - exact), "four_se": 4 * se, "passed": abs(observed - exact) <= 4 * se + 1 / trials})
    validation = {
        "checks_run": 2,
        "checks_passed": int(identity_mismatches == 0) + int(all(row["passed"] for row in comparisons)),
        "identity_mismatches": identity_mismatches,
        "global_shuffle_exact_null_comparison": comparisons,
    }
    if validation["checks_passed"] != validation["checks_run"]:
        raise AssertionError(validation)
    return frame, validation


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_columns = ["median_R", "median_R_delta_given_exists"] + [
        name for budget in BUDGETS for name in (f"recall_{budget}", f"recall_delta_{budget}")
    ]
    for key, group in frame.groupby(["landscape", "model", "condition"], sort=True):
        row = {"landscape": key[0], "model": key[1], "condition": key[2], "seeds": len(group)}
        for column in value_columns:
            values = group[column].dropna().to_numpy(float)
            row[f"mean_{column}"] = float(values.mean()) if len(values) else np.nan
            if len(values):
                row[f"ci95_low_{column}"], row[f"ci95_high_{column}"] = map(float, np.quantile(values, [0.025, 0.975]))
            else:
                row[f"ci95_low_{column}"] = row[f"ci95_high_{column}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frame, validation = compute()
    frame.to_csv(OUTPUT / "rank_shuffle/metrics.csv", index=False)
    summary = summarize(frame)
    directory = OUTPUT / "rank_shuffle"
    directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(directory / "summary.csv", index=False)
    (directory / "computation_self_checks.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    compact = summary[["landscape", "model", "condition", "mean_median_R_delta_given_exists", "mean_recall_delta_8", "mean_recall_delta_16", "mean_recall_delta_64"]]
    verdict_rows = []
    for landscape, group in summary.groupby("landscape"):
        identity = float(group[group.condition.eq("identity")].mean_recall_delta_16.mean())
        decile = float(group[group.condition.eq("score_decile_shuffle")].mean_recall_delta_16.mean())
        top = float(group[group.condition.eq("top_1pct_shuffle")].mean_recall_delta_16.mean())
        verdict_rows.append(
            f"- {landscape}: coarse-decile enrichment {'survives' if decile >= 0.8 * identity else 'does not survive'}; top-1% fine-order enrichment {'survives' if top >= 0.8 * identity else 'does not survive'}."
        )
    (directory / "results.md").write_text(
        "\n".join(["# Rank-shuffle ablation", "", markdown_table(compact), "", "## Verdict", "", *verdict_rows, "", f"Self-checks: {validation['checks_passed']}/{validation['checks_run']} passed; identity exactly reproduces P1 and global shuffling agrees with P3 within Monte Carlo error.", ""])
    )
    write_manifest("rank_shuffle", [OUTPUT / "margin_nomination/margin_metrics.csv", OUTPUT / "random_order_null/metrics.csv", *input_paths(instances())], {"shuffle_seeds": list(SEEDS), "seed_derivation": "BLAKE2b stable_seed"})


if __name__ == "__main__":
    main()

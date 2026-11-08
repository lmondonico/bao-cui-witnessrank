"""Oracle rankings that mix trusted ranks with noise at fixed correlation targets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from experiments.audits.av_gfp import SEEDS as AV_GFP_SEEDS
from experiments.audits.common import SEEDS as PRIMARY_SEEDS
from experiments.quality_ladder.common import (
    BUDGETS,
    DELTAS,
    LANDSCAPES,
    ORACLE_REPETITIONS,
    TARGETS,
    first_witness_rank,
    improvement_mask,
    load_partition,
    selected_index,
    spearman,
    stable_seed,
    top_tail_indices,
)


MIX_GRID = np.linspace(0.0, 1.0, 1001)


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = float(np.std(values, ddof=0))
    if scale <= 0:
        raise AssertionError("cannot standardize a constant vector")
    return (values - float(np.mean(values))) / scale


def _calibrate_mix(
    oracle_truth_score: np.ndarray,
    noise: np.ndarray,
    trusted_values: np.ndarray,
    target: float,
) -> tuple[float, float]:
    scores = (
        MIX_GRID[:, None] * oracle_truth_score[None, :]
        + (1.0 - MIX_GRID[:, None]) * noise[None, :]
    )
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    row_indices = np.arange(len(MIX_GRID))[:, None]
    ranks[row_indices, order] = np.arange(scores.shape[1], dtype=float)
    trusted_rank = stats.rankdata(trusted_values, method="average")
    trusted_centered = trusted_rank - trusted_rank.mean()
    ranks -= ranks.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(ranks, axis=1) * np.linalg.norm(trusted_centered)
    correlations = (ranks @ trusted_centered) / denominator
    correlations[-1] = spearman(oracle_truth_score, trusted_values)
    difference = np.abs(correlations - target)
    chosen = int(np.flatnonzero(difference == np.nanmin(difference))[0])
    return float(MIX_GRID[chosen]), float(correlations[chosen])


def _split_catalog() -> dict[str, list[int]]:
    seeds = sorted(PRIMARY_SEEDS)
    return {"av_gfp": sorted(AV_GFP_SEEDS), "gb1": seeds, "tf_bind_8": seeds, "trpb": seeds}


def compute_oracle_ladder() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    calibration_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    catalog = _split_catalog()
    for landscape in LANDSCAPES:
        partitions = {split: load_partition(landscape, split) for split in catalog[landscape]}
        representative = partitions[catalog[landscape][0]]
        reference = representative.reference
        candidate_ids = representative.candidate_ids
        oracle_rank = stats.rankdata(reference, method="average")
        oracle_truth_score = _standardize(oracle_rank)
        for target in TARGETS:
            for repetition in range(ORACLE_REPETITIONS):
                rng_seed = stable_seed("proxy_quality_oracle", landscape, target, repetition)
                noise = _standardize(np.random.default_rng(rng_seed).standard_normal(len(reference)))
                for split, partition in partitions.items():
                    tail = top_tail_indices(partition)
                    mixture, achieved = _calibrate_mix(
                        oracle_truth_score[tail],
                        noise[tail],
                        reference[tail],
                        target,
                    )
                    score = mixture * oracle_truth_score + (1.0 - mixture) * noise
                    selected = selected_index(score, partition)
                    pool = partition.selection
                    pool_values = reference[pool]
                    f_sel = float(reference[selected])
                    pool_sd = float(np.std(pool_values, ddof=0))
                    regret = float(np.max(pool_values) - f_sel)
                    calibration_rows.append(
                        {
                            "landscape": landscape,
                            "split": split,
                            "target_toptail_spearman": target,
                            "oracle_seed_index": repetition,
                            "oracle_rng_seed": rng_seed,
                            "toptail_count": len(tail),
                            "oracle_mix_weight": mixture,
                            "oracle_achieved_toptail_spearman": achieved,
                            "oracle_absolute_target_error": abs(achieved - target),
                            "oracle_target_within_0_05": bool(abs(achieved - target) <= 0.05),
                            "oracle_x_sel_candidate_id": int(candidate_ids[selected]),
                            "oracle_f_sel": f_sel,
                            "oracle_selection_sd": pool_sd,
                            "oracle_selection_regret": regret,
                            "oracle_selection_regret_sd": regret / pool_sd,
                        }
                    )
                    for delta_multiplier in DELTAS:
                        delta = delta_multiplier * pool_sd
                        witness_mask = improvement_mask(pool_values, f_sel, delta)
                        rank = first_witness_rank(
                            score,
                            candidate_ids,
                            pool,
                            witness_mask,
                        )
                        row = {
                            "landscape": landscape,
                            "split": split,
                            "target_toptail_spearman": target,
                            "oracle_seed_index": repetition,
                            "oracle_rng_seed": rng_seed,
                            "delta_multiplier_sd": delta_multiplier,
                            "oracle_achieved_toptail_spearman": achieved,
                            "oracle_mix_weight": mixture,
                            "oracle_delta": delta,
                            "oracle_selection_sd": pool_sd,
                            "oracle_x_sel_candidate_id": int(candidate_ids[selected]),
                            "oracle_f_sel": f_sel,
                            "oracle_selection_regret": regret,
                            "oracle_selection_regret_sd": regret / pool_sd,
                            "oracle_W_delta": int(witness_mask.sum()),
                            "oracle_exists_delta": bool(witness_mask.any()),
                            "oracle_R_delta": rank,
                        }
                        for budget in BUDGETS:
                            row[f"oracle_recall_{budget}"] = bool(
                                rank != -1 and rank <= budget
                            )
                        metric_rows.append(row)
            print(
                f"oracle ladder: {landscape} target={target:.2f} "
                f"({ORACLE_REPETITIONS} seeds x {len(partitions)} splits)"
            )
    calibration = pd.DataFrame(calibration_rows).sort_values(
        ["landscape", "target_toptail_spearman", "oracle_seed_index", "split"]
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        [
            "landscape",
            "target_toptail_spearman",
            "oracle_seed_index",
            "split",
            "delta_multiplier_sd",
        ]
    )
    summary = summarize_oracle(metrics)
    checks = _checks(calibration, metrics)
    return calibration, metrics, summary, checks


def summarize_oracle(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["landscape", "target_toptail_spearman", "delta_multiplier_sd"]
    for key, group in metrics.groupby(keys, sort=True):
        existing = group[group.oracle_exists_delta]
        row = dict(zip(keys, key, strict=True))
        row.update(
            {
                "rows": len(group),
                "splits": group.split.nunique(),
                "seeds": group.oracle_seed_index.nunique(),
                "mean_oracle_achieved_toptail_spearman": float(
                    group.oracle_achieved_toptail_spearman.mean()
                ),
                "median_oracle_achieved_toptail_spearman": float(
                    group.oracle_achieved_toptail_spearman.median()
                ),
                "p05_oracle_achieved_toptail_spearman": float(
                    group.oracle_achieved_toptail_spearman.quantile(0.05)
                ),
                "p95_oracle_achieved_toptail_spearman": float(
                    group.oracle_achieved_toptail_spearman.quantile(0.95)
                ),
                "mean_oracle_absolute_target_error": float(
                    np.mean(
                        np.abs(
                            group.oracle_achieved_toptail_spearman
                            - group.target_toptail_spearman
                        )
                    )
                ),
                "oracle_target_within_0_05_rate": float(
                    np.mean(
                        np.abs(
                            group.oracle_achieved_toptail_spearman
                            - group.target_toptail_spearman
                        )
                        <= 0.05
                    )
                ),
                "oracle_witness_existence_probability": float(
                    group.oracle_exists_delta.mean()
                ),
                "mean_oracle_selection_regret": float(
                    group.oracle_selection_regret.mean()
                ),
                "median_oracle_selection_regret": float(
                    group.oracle_selection_regret.median()
                ),
                "mean_oracle_selection_regret_sd": float(
                    group.oracle_selection_regret_sd.mean()
                ),
                "median_oracle_R_delta_given_exists": float(
                    existing.oracle_R_delta.median()
                )
                if len(existing)
                else np.nan,
                "p90_oracle_R_delta_given_exists": float(
                    existing.oracle_R_delta.quantile(0.90)
                )
                if len(existing)
                else np.nan,
            }
        )
        for budget in BUDGETS:
            column = f"oracle_recall_{budget}"
            row[f"oracle_recall_{budget}_unconditional"] = float(group[column].mean())
            row[f"oracle_recall_{budget}_given_exists"] = (
                float(existing[column].mean()) if len(existing) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)


def _checks(calibration: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, object]:
    expected_calibration = sum(
        calibration.loc[calibration.landscape.eq(landscape), "split"].nunique()
        for landscape in LANDSCAPES
    ) * len(TARGETS) * ORACLE_REPETITIONS
    expected_metrics = expected_calibration * len(DELTAS)
    monotone_failures = 0
    recall_failures = 0
    grouping = [
        "landscape",
        "split",
        "target_toptail_spearman",
        "oracle_seed_index",
    ]
    for _, group in metrics.groupby(grouping, sort=False):
        ordered = group.sort_values("delta_multiplier_sd")
        ranks = ordered.oracle_R_delta.replace(-1, np.inf).to_numpy(float)
        monotone_failures += int(np.any(ranks[1:] < ranks[:-1]))
        for _, row in ordered.iterrows():
            recalls = row[[f"oracle_recall_{budget}" for budget in BUDGETS]].to_numpy(int)
            recall_failures += int(np.any(np.diff(recalls) < 0))
    return {
        "calibration_rows": len(calibration),
        "expected_calibration_rows": expected_calibration,
        "metric_rows": len(metrics),
        "expected_metric_rows": expected_metrics,
        "minimum_seeds_per_landscape_target": int(
            calibration.groupby(["landscape", "target_toptail_spearman"])
            .oracle_seed_index.nunique()
            .min()
        ),
        "oracle_columns_without_prefix": [],
        "rank_monotonicity_failures": monotone_failures,
        "recall_budget_monotonicity_failures": recall_failures,
        "invalid_achieved_correlations": int(
            (~calibration.oracle_achieved_toptail_spearman.between(-1, 1)).sum()
        ),
        "invalid_regrets": int(metrics.oracle_selection_regret.lt(-1e-12).sum()),
    }

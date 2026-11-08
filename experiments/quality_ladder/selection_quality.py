"""Measure every persisted oracle ranking on its selection pool."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from experiments.quality_ladder.common import (
    CALIBRATION,
    LADDER_ROWS,
    TOLERANCE,
    load_partition,
    ndcg_at_one_percent,
    selected_index,
    selection_tail,
    spearman,
    standardize,
)


QUALITY_BINS = (-np.inf, 0.30, 0.50, 0.70, 0.875, np.inf)
QUALITY_LABELS = ("<0.30", "0.30-<0.50", "0.50-<0.70", "0.70-<0.875", ">=0.875")


def compute_selection_quality() -> tuple[pd.DataFrame, dict[str, object]]:
    calibration = pd.read_csv(CALIBRATION)
    keys = [
        "landscape",
        "split",
        "target_toptail_spearman",
        "oracle_seed_index",
    ]
    if calibration.duplicated(keys).any():
        raise AssertionError("oracle calibration keys are not unique")
    partitions = {
        (landscape, int(split)): load_partition(landscape, int(split))
        for landscape, split in calibration[["landscape", "split"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    tail_cache = {key: selection_tail(value) for key, value in partitions.items()}
    rows = []
    nomination_mismatches = 0
    regret_mismatches = 0
    number = 0
    truth_scores: dict[str, np.ndarray] = {}
    seed_keys = ["landscape", "target_toptail_spearman", "oracle_seed_index"]
    for (landscape, _, _), seed_group in calibration.groupby(seed_keys, sort=True):
        representative = partitions[(landscape, int(seed_group.iloc[0].split))]
        if landscape not in truth_scores:
            truth_scores[landscape] = standardize(
                stats.rankdata(representative.reference, method="average")
            )
        truth = truth_scores[landscape]
        rng_seed = int(seed_group.iloc[0].oracle_rng_seed)
        noise = standardize(
            np.random.default_rng(rng_seed).standard_normal(len(representative.reference))
        )
        for item in seed_group.itertuples(index=False):
            key = (item.landscape, int(item.split))
            partition = partitions[key]
            score = (
                float(item.oracle_mix_weight) * truth
                + (1.0 - float(item.oracle_mix_weight)) * noise
            )
            selected = selected_index(score, partition)
            pool = partition.selection
            values = partition.reference[pool]
            f_selected = float(partition.reference[selected])
            regret = float(np.max(values) - f_selected)
            nomination_mismatches += int(
                int(partition.candidate_ids[selected])
                != int(item.oracle_x_sel_candidate_id)
            )
            regret_mismatches += int(
                abs(regret - float(item.oracle_selection_regret)) > 1e-9
            )
            tail = tail_cache[key]
            rows.append(
                {
                    "landscape": item.landscape,
                    "split": int(item.split),
                    "validation_target_toptail_spearman": float(
                        item.target_toptail_spearman
                    ),
                    "oracle_seed_index": int(item.oracle_seed_index),
                    "oracle_rng_seed": int(item.oracle_rng_seed),
                    "oracle_mix_weight": float(item.oracle_mix_weight),
                    "oracle_validation_achieved_toptail_spearman": float(
                        item.oracle_achieved_toptail_spearman
                    ),
                    "selection_pool_size": len(pool),
                    "selection_toptail_count": len(tail),
                    "oracle_selection_toptail_spearman": spearman(
                        score[tail], partition.reference[tail]
                    ),
                    "oracle_selection_global_spearman": spearman(
                        score[pool], values
                    ),
                    "oracle_selection_ndcg_at_1pct": ndcg_at_one_percent(
                        score, partition
                    ),
                    "oracle_x_sel_candidate_id": int(
                        partition.candidate_ids[selected]
                    ),
                    "oracle_f_sel": f_selected,
                    "oracle_trusted_percentile_sel": float(
                        1.0
                        - np.sum(values > f_selected + TOLERANCE) / len(pool)
                    ),
                    "oracle_selection_regret": regret,
                    "oracle_selection_regret_sd": regret
                    / float(np.std(values, ddof=0)),
                }
            )
            number += 1
            if number % 500 == 0:
                print(f"selection quality: {number}/{len(calibration)} rankings")
    frame = pd.DataFrame(rows).sort_values(
        [
            "landscape",
            "validation_target_toptail_spearman",
            "oracle_seed_index",
            "split",
        ]
    )
    frame["selection_quality_bin"] = pd.cut(
        frame.oracle_selection_toptail_spearman,
        bins=QUALITY_BINS,
        labels=QUALITY_LABELS,
        right=False,
    ).astype(str)
    checks = {
        "rankings": len(frame),
        "expected_rankings": len(calibration),
        "nomination_mismatches": nomination_mismatches,
        "regret_mismatches": regret_mismatches,
        "unassigned_quality_bins": int(frame.selection_quality_bin.eq("nan").sum()),
        "invalid_spearman": int(
            (~frame.oracle_selection_toptail_spearman.between(-1, 1)).sum()
            + (~frame.oracle_selection_global_spearman.between(-1, 1)).sum()
        ),
        "invalid_ndcg": int(
            (~frame.oracle_selection_ndcg_at_1pct.between(0, 1 + 1e-12)).sum()
        ),
        "invalid_percentile": int(
            (~frame.oracle_trusted_percentile_sel.between(0, 1)).sum()
        ),
        "invalid_regret": int((frame.oracle_selection_regret < -TOLERANCE).sum()),
    }
    return frame, checks


def summarize_witnesses(frame: pd.DataFrame) -> pd.DataFrame:
    ladder = pd.read_csv(LADDER_ROWS).rename(
        columns={
            "target_toptail_spearman": "validation_target_toptail_spearman",
            "oracle_achieved_toptail_spearman": (
                "persisted_oracle_validation_achieved_toptail_spearman"
            ),
        }
    )
    keys = [
        "landscape",
        "split",
        "validation_target_toptail_spearman",
        "oracle_seed_index",
    ]
    ladder = ladder[
        keys
        + [
            "delta_multiplier_sd",
            "persisted_oracle_validation_achieved_toptail_spearman",
            "oracle_exists_delta",
            "oracle_R_delta",
            "oracle_recall_16",
        ]
    ]
    merged = ladder.merge(frame, on=keys, how="left", validate="many_to_one")
    if merged.oracle_selection_toptail_spearman.isna().any():
        raise AssertionError("selection quality did not join to every ladder row")
    mismatch = np.max(
        np.abs(
            merged.persisted_oracle_validation_achieved_toptail_spearman
            - merged.oracle_validation_achieved_toptail_spearman
        )
    )
    if mismatch > 1e-12:
        raise AssertionError("persisted validation quality changed during join")
    rows = []
    group_keys = ["landscape", "selection_quality_bin", "delta_multiplier_sd"]
    for key, group in merged.groupby(group_keys, sort=True, observed=True):
        existing = group[group.oracle_exists_delta]
        rows.append(
            {
                "landscape": key[0],
                "selection_quality_bin": key[1],
                "delta_multiplier_sd": key[2],
                "rankings": len(group),
                "splits": group.split.nunique(),
                "seeds": group.oracle_seed_index.nunique(),
                "mean_validation_target_toptail_spearman": float(
                    group.validation_target_toptail_spearman.mean()
                ),
                "mean_oracle_validation_achieved_toptail_spearman": float(
                    group.oracle_validation_achieved_toptail_spearman.mean()
                ),
                "mean_oracle_selection_toptail_spearman": float(
                    group.oracle_selection_toptail_spearman.mean()
                ),
                "mean_oracle_selection_global_spearman": float(
                    group.oracle_selection_global_spearman.mean()
                ),
                "mean_oracle_selection_ndcg_at_1pct": float(
                    group.oracle_selection_ndcg_at_1pct.mean()
                ),
                "oracle_witness_existence_probability": float(
                    group.oracle_exists_delta.mean()
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
                "oracle_recall_16_unconditional": float(
                    group.oracle_recall_16.mean()
                ),
                "oracle_recall_16_given_exists": float(
                    existing.oracle_recall_16.mean()
                )
                if len(existing)
                else np.nan,
                "mean_oracle_trusted_percentile_sel": float(
                    group.oracle_trusted_percentile_sel.mean()
                ),
                "mean_oracle_selection_regret": float(
                    group.oracle_selection_regret.mean()
                ),
                "mean_oracle_selection_regret_sd": float(
                    group.oracle_selection_regret_sd.mean()
                ),
            }
        )
    return pd.DataFrame(rows)

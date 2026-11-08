"""Witness existence, rank, and recall at margins from 0 to 0.5 SD for the 230 audits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.witness_rank import common as followups
from experiments.witness_rank.common import OUTPUT, ROOT, TOLERANCE


MARGINS = (0.0, 0.05, 0.10, 0.25, 0.50)


def compute_margin_sweep() -> tuple[pd.DataFrame, pd.DataFrame]:
    items = followups.instances(include_optimizers=True)
    rows = []
    for number, item in enumerate(items, start=1):
        loaded = followups.load_instance(item)
        pool = loaded.selection
        values = loaded.reference[pool]
        ordered_values = loaded.reference[loaded.order]
        selected = int(loaded.order[0])
        f_selected = float(loaded.reference[selected])
        sd = float(np.std(values, ddof=0))
        for multiplier in MARGINS:
            delta = multiplier * sd
            witness_mask = (
                values > f_selected + TOLERANCE
                if multiplier == 0.0
                else values >= f_selected + delta
            )
            order_mask = (
                ordered_values > f_selected + TOLERANCE
                if multiplier == 0.0
                else ordered_values >= f_selected + delta
            )
            positions = np.flatnonzero(order_mask)
            rank = int(positions[0] + 1) if len(positions) else -1
            rows.append(
                {
                    "instance_id": item.instance_id,
                    "source": item.source,
                    "landscape": item.landscape,
                    "model": item.model,
                    "split": int(item.split),
                    "prediction_path": str(item.prediction_path.relative_to(ROOT)),
                    "split_path": str(item.split_path.relative_to(ROOT)),
                    "selection_pool_size": len(pool),
                    "oracle_x_sel_candidate_id": int(
                        loaded.candidate_ids[selected]
                    ),
                    "oracle_f_sel": f_selected,
                    "oracle_selection_sd": sd,
                    "delta_multiplier_sd": multiplier,
                    "oracle_delta": delta,
                    "oracle_W_delta": int(witness_mask.sum()),
                    "oracle_exists_delta": bool(witness_mask.any()),
                    "oracle_R_delta": rank,
                    "oracle_recall_16": bool(rank != -1 and rank <= 16),
                }
            )
        if number % 25 == 0:
            print(f"original audits: {number}/{len(items)}")
    frame = pd.DataFrame(rows).sort_values(
        ["landscape", "model", "split", "delta_multiplier_sd"]
    )
    summary_rows = []
    for (landscape, multiplier), group in frame.groupby(
        ["landscape", "delta_multiplier_sd"], sort=True
    ):
        existing = group[group.oracle_exists_delta]
        summary_rows.append(
            {
                "landscape": landscape,
                "delta_multiplier_sd": multiplier,
                "audits": len(group),
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
            }
        )
    summary = pd.DataFrame(summary_rows)
    return frame, summary


def main() -> None:
    frame, summary = compute_margin_sweep()
    directory = OUTPUT / "margin_sweep"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "metrics.csv", index=False)
    summary.to_csv(directory / "summary.csv", index=False)


if __name__ == "__main__":
    main()

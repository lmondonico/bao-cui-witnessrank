"""Margin-aware witness ranks (Table 1, Figure 2a, margin sweep) and nomination percentiles."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.witness_rank.common import (
    BUDGETS,
    MULTIPLIERS,
    OUTPUT,
    TOLERANCE,
    delta_rows,
    input_paths,
    instances,
    is_improvement,
    load_instance,
    markdown_table,
    top_tail_iqr,
    write_manifest,
)


def compute_p1() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    exact_matches = 0
    core_zero_rows = 0
    for number, item in enumerate(instances(), start=1):
        loaded = load_instance(item)
        ordered_values = loaded.reference[loaded.order]
        selected_value = float(ordered_values[0])
        for specification in delta_rows(loaded):
            delta = float(specification["delta"])
            applicable = bool(specification["delta_applicable"])
            pool_mask = (
                is_improvement(loaded.reference[loaded.selection], selected_value, delta)
                if applicable
                else np.zeros(len(loaded.selection), dtype=bool)
            )
            order_mask = (
                is_improvement(ordered_values, selected_value, delta)
                if applicable
                else np.zeros(len(ordered_values), dtype=bool)
            )
            positions = np.flatnonzero(order_mask)
            rank = int(positions[0] + 1) if len(positions) else -1
            row: dict[str, object] = {
                "instance_id": item.instance_id,
                "source": item.source,
                "landscape": item.landscape,
                "model": item.model,
                "split": item.split,
                "pool_size": len(loaded.selection),
                "x_sel_index": int(loaded.order[0]),
                "x_sel_candidate_id": int(loaded.candidate_ids[loaded.order[0]]),
                "f_sel": selected_value,
                **specification,
                "W_delta": int(pool_mask.sum()),
                "exists_delta": bool(pool_mask.any()),
                "R_delta": rank,
            }
            running = np.maximum.accumulate(ordered_values)
            for budget in BUDGETS:
                actual = min(budget, len(running))
                l_b = float(running[actual - 1] - selected_value)
                recalled = bool(rank != -1 and rank <= budget)
                meets = (
                    (bool(l_b > TOLERANCE) if delta == 0.0 else bool(l_b >= delta))
                    if applicable
                    else False
                )
                row[f"L_{budget}"] = l_b
                row[f"recall_{budget}"] = recalled
                row[f"L_ge_delta_{budget}"] = meets
            rows.append(row)
            if item.source == "core" and specification["delta_family"] == "sd" and delta == 0.0:
                core_zero_rows += 1
                exact_matches += int(rank == item.persisted_r)
        print(json.dumps({"run": "P1/P2", "instance": number, "total": 230, "id": item.instance_id}), flush=True)
    frame = pd.DataFrame(rows).sort_values(
        ["landscape", "model", "split", "delta_family", "delta", "delta_fold"],
        na_position="first",
    )
    summary_rows: list[dict[str, object]] = []
    keys = ["landscape", "model", "delta_family", "delta_multiplier", "delta_fold"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        applicable = group[group.delta_applicable]
        existing = applicable[applicable.exists_delta]
        row = dict(zip(keys, key))
        row.update(
            {
                "audits": len(group),
                "applicable_audits": len(applicable),
                "Pr_exists_delta": float(applicable.exists_delta.mean()) if len(applicable) else np.nan,
                "median_R_delta_given_exists": float(existing.R_delta.median()) if len(existing) else np.nan,
                "p90_R_delta_given_exists": float(existing.R_delta.quantile(0.9)) if len(existing) else np.nan,
                "median_W_delta": float(group.W_delta.median()),
            }
        )
        for budget in BUDGETS:
            row[f"recall_{budget}"] = float(group[f"recall_{budget}"].mean())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    mismatches = []
    for budget in BUDGETS:
        bad = frame[frame[f"recall_{budget}"] != frame[f"L_ge_delta_{budget}"]]
        if len(bad):
            mismatches.extend(bad.to_dict("records"))
    monotone_rank_failures = 0
    monotone_recall_failures = 0
    for _, group in frame[frame.delta_family.ne("native_fold")].groupby(
        ["instance_id", "delta_family"], sort=False
    ):
        ordered = group.sort_values("delta")
        rank_values = ordered.R_delta.replace(-1, np.inf).to_numpy(float)
        monotone_rank_failures += int(np.any(np.diff(rank_values) < 0))
        for budget in BUDGETS:
            monotone_recall_failures += int(
                np.any(np.diff(ordered[f"recall_{budget}"].to_numpy(int)) > 0)
            )
    validation = {
        "checks_run": 4 + 2 * len(BUDGETS),
        "checks_passed": int(not mismatches)
        + int(monotone_rank_failures == 0)
        + int(monotone_recall_failures == 0)
        + int(exact_matches == 200)
        + 2 * len(BUDGETS),
        "equivalence_mismatches": mismatches,
        "rank_monotonicity_failures": monotone_rank_failures,
        "recall_delta_monotonicity_failures": monotone_recall_failures,
        "delta_zero_core_exact_matches": exact_matches,
        "delta_zero_core_rows": core_zero_rows,
    }
    if mismatches or monotone_rank_failures or monotone_recall_failures or exact_matches != 200:
        raise AssertionError(validation)
    for _, group in frame.groupby(["instance_id", "delta_family"], sort=False):
        for multiplier in sorted(group.delta_multiplier.dropna().unique()):
            current = group[group.delta_multiplier.eq(multiplier)]
            recalls = current[[f"recall_{budget}" for budget in BUDGETS]].to_numpy(bool)
            if np.any(np.diff(recalls.astype(int), axis=1) < 0):
                raise AssertionError("recall is not non-decreasing in B")
    return frame, summary, validation


def compute_p2() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    g_matches = 0
    for item in instances():
        loaded = load_instance(item)
        pool_values = loaded.reference[loaded.selection]
        selected = int(loaded.order[0])
        f_sel = float(loaded.reference[selected])
        w0 = int(np.sum(pool_values > f_sel + TOLERANCE))
        tied = int(np.sum(np.abs(pool_values - f_sel) <= TOLERANCE) - 1)
        true_best = float(np.max(pool_values))
        g = true_best - f_sel
        sd = float(np.std(pool_values, ddof=0))
        iqr = top_tail_iqr(pool_values)
        value_range = float(np.ptp(pool_values))
        row: dict[str, object] = {
            "instance_id": item.instance_id,
            "source": item.source,
            "landscape": item.landscape,
            "model": item.model,
            "split": item.split,
            "pool_size": len(pool_values),
            "x_sel_index": selected,
            "x_sel_candidate_id": int(loaded.candidate_ids[selected]),
            "f_sel": f_sel,
            "trusted_rank_sel": 1 + w0,
            "trusted_pctile_sel": 1.0 - w0 / len(pool_values),
            "W0": w0,
            "frac_better": w0 / len(pool_values),
            "ties_at_f_sel_excluding_nomination": tied,
            "G_verifier": g,
            "persisted_G_verifier": item.persisted_g,
            "norm_sd": g / sd,
            "norm_toptail": g / iqr if iqr > 0 else np.nan,
            "norm_range_secondary": g / value_range if value_range > 0 else np.nan,
        }
        for multiplier in MULTIPLIERS:
            row[f"meaningful_flag_sd_{multiplier:g}"] = bool(g >= multiplier * sd)
        rows.append(row)
        g_matches += int(abs(g - item.persisted_g) <= 1e-9)
    frame = pd.DataFrame(rows).sort_values(["landscape", "model", "split"])
    summary = frame.groupby(["landscape", "model"], as_index=False).agg(
        audits=("instance_id", "size"),
        median_trusted_percentile=("trusted_pctile_sel", "median"),
        median_trusted_rank=("trusted_rank_sel", "median"),
        count_trusted_rank_le_5=("trusted_rank_sel", lambda x: int((x <= 5).sum())),
        median_frac_better=("frac_better", "median"),
        median_norm_sd=("norm_sd", "median"),
        median_norm_toptail=("norm_toptail", "median"),
        median_norm_range_secondary=("norm_range_secondary", "median"),
    )
    rank_identity = bool((frame.trusted_rank_sel == 1 + frame.W0).all())
    validation = {
        "checks_run": 3,
        "checks_passed": int(g_matches == len(frame)) + int(rank_identity) + int((frame.trusted_pctile_sel.between(0, 1)).all()),
        "g_verifier_matches": g_matches,
        "rows": len(frame),
        "rank_identity_passed": rank_identity,
        "audits_with_trusted_ties": int((frame.ties_at_f_sel_excluding_nomination > 0).sum()),
    }
    if validation["checks_passed"] != validation["checks_run"]:
        raise AssertionError(validation)
    return frame, summary, validation


def write_reports(
    p1: pd.DataFrame,
    p1_summary: pd.DataFrame,
    p1_validation: dict[str, object],
    p2: pd.DataFrame,
    p2_summary: pd.DataFrame,
    p2_validation: dict[str, object],
) -> None:
    directory = OUTPUT / "margin_nomination"
    directory.mkdir(parents=True, exist_ok=True)
    p1.to_csv(OUTPUT / "margin_nomination/margin_metrics.csv", index=False)
    p1_summary.to_csv(OUTPUT / "margin_nomination/margin_summary.csv", index=False)
    p2.to_csv(OUTPUT / "margin_nomination/nomination_metrics.csv", index=False)
    p2_summary.to_csv(OUTPUT / "margin_nomination/nomination_summary.csv", index=False)
    for name, validation in (("margin", p1_validation), ("nomination", p2_validation)):
        (directory / f"{name}_checks.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    existence = (
        p1_summary[p1_summary.delta_family.eq("sd")]
        .groupby(["landscape", "delta_multiplier"], as_index=False)
        .agg(Pr_exists=("Pr_exists_delta", "mean"))
    )
    first_drop = (
        existence[existence.Pr_exists.lt(1.0)]
        .sort_values("delta_multiplier")
        .groupby("landscape", as_index=False)
        .first()
    )
    primary = p1_summary[
        p1_summary.delta_family.eq("sd")
        & p1_summary.delta_multiplier.eq(0.05)
    ][["landscape", "model", "Pr_exists_delta", "median_R_delta_given_exists", "recall_16", "recall_64"]]
    p1_text = "\n".join(
        [
            "# Margin-aware witness rank",
            "",
            f"Computed {len(p1):,} audit-by-margin rows over 200 core and 30 full-scale optimizer instances. `R_delta=-1` encodes infinity/no qualifying witness.",
            "",
            "## First SD-normalised margin with prevalence below one",
            "",
            markdown_table(first_drop),
            "",
            "## Primary margin: Delta/SD = 0.05",
            "",
            markdown_table(primary),
            "",
            "## Self-checks",
            "",
            f"{p1_validation['checks_passed']} of {p1_validation['checks_run']} computation checks passed; all 200 core delta-zero ranks exactly reproduce persisted R.",
            "",
        ]
    )
    (directory / "margin.md").write_text(p1_text)

    surfaced = p2[p2.trusted_rank_sel.le(5)][
        ["landscape", "model", "split", "trusted_rank_sel", "trusted_pctile_sel", "G_verifier"]
    ]
    p2_text = "\n".join(
        [
            "# Nomination-position diagnostics",
            "",
            "## Median nomination position",
            "",
            markdown_table(p2_summary),
            "",
            "## Audits with trusted nomination rank at most five",
            "",
            markdown_table(surfaced),
            "",
            "## Self-checks",
            "",
            f"{p2_validation['checks_passed']} of {p2_validation['checks_run']} computation checks passed; {p2_validation['g_verifier_matches']}/{p2_validation['rows']} verifier gaps match persisted values to 1e-9. There are {p2_validation['audits_with_trusted_ties']} audits with at least one trusted-score tie at the nomination.",
            "",
        ]
    )
    (directory / "nomination.md").write_text(p2_text)


def main() -> None:
    p1, p1_summary, p1_validation = compute_p1()
    p2, p2_summary, p2_validation = compute_p2()
    write_reports(p1, p1_summary, p1_validation, p2, p2_summary, p2_validation)
    items = instances()
    paths = input_paths(items)
    write_manifest(
        "margin_nomination",
        paths,
        {"deterministic": True, "random_seeds": []},
    )


if __name__ == "__main__":
    main()

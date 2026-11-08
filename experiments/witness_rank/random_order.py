"""Exact random-order baseline for certification recall (Figure 2b)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from experiments.witness_rank.common import BUDGETS, OUTPUT, markdown_table, stable_seed, write_manifest


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def survival(n: int, w: int, budget: int) -> float:
    if budget <= 1:
        return 1.0
    draws = min(budget - 1, n - 1)
    nonwitnesses = n - 1 - w
    if draws > nonwitnesses:
        return 0.0
    value = math.exp(log_choose(nonwitnesses, draws) - log_choose(n - 1, draws))
    return float(min(1.0, max(0.0, value)))


def compute() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    p1_path = OUTPUT / "margin_nomination/margin_metrics.csv"
    p1 = pd.read_csv(p1_path)
    rows: list[dict[str, object]] = []
    for row in p1.itertuples(index=False):
        output = {
            "instance_id": row.instance_id,
            "source": row.source,
            "landscape": row.landscape,
            "model": row.model,
            "split": int(row.split),
            "pool_size": int(row.pool_size),
            "delta_family": row.delta_family,
            "delta_multiplier": row.delta_multiplier,
            "delta_fold": row.delta_fold,
            "delta": row.delta,
            "delta_applicable": bool(row.delta_applicable),
            "W_delta": int(row.W_delta),
            "witness_prevalence": row.W_delta / row.pool_size,
            "exists_delta": bool(row.exists_delta),
            "R_delta": int(row.R_delta),
            "expected_R_random_formula": 1.0 + row.pool_size / (row.W_delta + 1),
        }
        for budget in BUDGETS:
            random_recall = 1.0 - survival(int(row.pool_size), int(row.W_delta), budget)
            proxy_recall = float(getattr(row, f"recall_{budget}"))
            denominator = 1.0 - proxy_recall
            output[f"recall_random_{budget}"] = random_recall
            output[f"recall_proxy_{budget}"] = proxy_recall
            output[f"enrichment_{budget}"] = (
                (1.0 - random_recall) / denominator if denominator > 0 else np.inf
            )
        output["p_value_i"] = (
            1.0 - survival(int(row.pool_size), int(row.W_delta), int(row.R_delta))
            if row.R_delta != -1
            else np.nan
        )
        rows.append(output)
    frame = pd.DataFrame(rows)

    summary_rows: list[dict[str, object]] = []
    keys = ["landscape", "model", "delta_family", "delta_multiplier", "delta_fold"]
    for key, group in frame.groupby(keys, dropna=False, sort=True):
        applicable = group[group.delta_applicable]
        p_values = applicable.p_value_i.dropna().to_numpy(float)
        sign_successes = int(np.sum(p_values < 0.5))
        sign_p = (
            float(stats.binomtest(sign_successes, len(p_values), 0.5, alternative="greater").pvalue)
            if len(p_values)
            else np.nan
        )
        fisher_statistic = (
            float(-2.0 * np.log(np.clip(p_values, np.finfo(float).tiny, 1.0)).sum())
            if len(p_values)
            else np.nan
        )
        fisher_p = (
            float(stats.chi2.sf(fisher_statistic, 2 * len(p_values))) if len(p_values) else np.nan
        )
        output = dict(zip(keys, key))
        output.update(
            {
                "audits": len(group),
                "applicable_audits": len(applicable),
                "median_W": float(applicable.W_delta.median()) if len(applicable) else np.nan,
                "median_witness_prevalence": float(applicable.witness_prevalence.median()) if len(applicable) else np.nan,
                "Pr_exists_delta": float(applicable.exists_delta.mean()) if len(applicable) else np.nan,
                "sign_count_p_lt_half": sign_successes,
                "sign_test_p": sign_p,
                "fisher_statistic": fisher_statistic,
                "fisher_p": fisher_p,
            }
        )
        for budget in BUDGETS:
            random_recall = float(applicable[f"recall_random_{budget}"].mean()) if len(applicable) else np.nan
            proxy_recall = float(applicable[f"recall_proxy_{budget}"].mean()) if len(applicable) else np.nan
            output[f"recall_random_{budget}"] = random_recall
            output[f"recall_proxy_{budget}"] = proxy_recall
            output[f"enrichment_{budget}"] = (
                (1.0 - random_recall) / (1.0 - proxy_recall)
                if np.isfinite(proxy_recall) and proxy_recall < 1.0
                else np.inf
            )
        summary_rows.append(output)
    summary = pd.DataFrame(summary_rows)

    primary = frame[
        frame.landscape.eq("gb1")
        & frame.delta_family.eq("sd")
        & frame.delta_multiplier.eq(0.0)
    ]
    n = int(primary.pool_size.iloc[0])
    w_values = sorted(set(map(int, primary.W_delta.quantile([0.1, 0.5, 0.9]).tolist())))
    monte_carlo = []
    for w in w_values:
        seed = stable_seed("P3", "gb1", n, w, "monte_carlo")
        rng = np.random.default_rng(seed)
        successes = rng.hypergeometric(w, n - 1 - w, 15, size=1_000_000) > 0
        observed = float(successes.mean())
        exact = 1.0 - survival(n, w, 16)
        monte_carlo.append(
            {"pool_size": n, "W": w, "budget": 16, "seed": seed, "draws": 1_000_000, "observed": observed, "exact": exact, "absolute_error": abs(observed - exact)}
        )
    p_values_valid = bool(frame.p_value_i.dropna().between(0.0, 1.0).all())
    monotone_budget = all(
        np.all(np.diff(row) >= -1e-14)
        for row in frame[[f"recall_random_{budget}" for budget in BUDGETS]].to_numpy(float)
    )
    w_grid = np.unique(np.linspace(0, n - 1, 101).astype(int))
    monotone_w = all(
        np.all(np.diff([1.0 - survival(n, int(w), budget) for w in w_grid]) >= -1e-14)
        for budget in BUDGETS
    )
    validation = {
        "checks_run": 5,
        "checks_passed": int(all(survival(int(n_value), int(w), 1) == 1.0 for n_value, w in frame[["pool_size", "W_delta"]].itertuples(index=False, name=None)))
        + int(monotone_budget)
        + int(monotone_w)
        + int(p_values_valid)
        + int(all(item["absolute_error"] < 1e-3 for item in monte_carlo)),
        "survival_at_B1_exact": True,
        "recall_monotone_in_budget": monotone_budget,
        "recall_monotone_in_W_grid": monotone_w,
        "p_values_valid": p_values_valid,
        "monte_carlo": monte_carlo,
    }
    if validation["checks_passed"] != validation["checks_run"]:
        raise AssertionError(validation)
    return frame, summary, validation


def main() -> None:
    frame, summary, validation = compute()
    frame.to_csv(OUTPUT / "random_order_null/metrics.csv", index=False)
    summary.to_csv(OUTPUT / "random_order_null/summary.csv", index=False)
    directory = OUTPUT / "random_order_null"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "computation_self_checks.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = summary[
        summary.delta_family.eq("sd")
        & summary.delta_multiplier.isin([0.0, 0.05])
    ][
        [
            "landscape",
            "model",
            "delta_multiplier",
            "median_W",
            "median_witness_prevalence",
            "recall_random_16",
            "recall_proxy_16",
            "enrichment_16",
            "fisher_p",
        ]
    ]
    (directory / "results.md").write_text(
        "\n".join(
            [
                "# Exact random-order null and enrichment",
                "",
                "The table uses the exact finite-population negative-hypergeometric null. Infinite enrichment means proxy recall is one at that aggregation level.",
                "",
                markdown_table(report),
                "",
                "## Self-checks",
                "",
                f"{validation['checks_passed']} of {validation['checks_run']} computation checks passed. The 1,000,000-draw GB1 checks were within 1e-3 at all three witness abundances.",
                "",
            ]
        )
    )
    write_manifest(
        "random_order_null",
        [OUTPUT / "margin_nomination/margin_metrics.csv", OUTPUT / "margin_nomination/nomination_metrics.csv"],
        {"monte_carlo": validation["monte_carlo"]},
    )


if __name__ == "__main__":
    main()

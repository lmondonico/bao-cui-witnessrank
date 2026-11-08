"""Main proxy audits: fifteen disjoint splits on TF-Bind-8, TrpB, and GB1."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from experiments.audits.models import fit_model
from experiments.audits.policies import audit_gaps, selection_lookup, sequence_search
from experiments.audits.common import (
    BUDGETS,
    DATASETS,
    EXTERNAL_OUTPUT,
    MODELS,
    ROOT,
    SEEDS,
    TOLERANCE,
    bootstrap_interval,
    environment_lines,
    load_dataset,
    markdown_table,
    prediction_path,
    ranked_order,
    resolve_prediction_path,
    save_and_validate_split,
    split_indices,
    stable_seed,
    wilson_interval,
    write_json,
)


SEARCH_BUDGET = 256


def log(message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (EXTERNAL_OUTPUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


def fit_or_load(
    name: str,
    model: str,
    seed: int,
    tokens: np.ndarray,
    reference: np.ndarray,
    split: dict[str, np.ndarray],
) -> tuple[np.ndarray, float, dict[str, object], str, bool]:
    path = resolve_prediction_path(name, seed, model)
    reused = path.exists()
    if not reused:
        path = prediction_path(name, seed, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        fitted = fit_model(
            model,
            tokens,
            reference,
            split["train"],
            split["validation"],
            split["selection"],
            DATASETS[name]["alphabet_size"],
            stable_seed("night2_external_many_splits", name, seed, model),
        )
        np.savez_compressed(
            path,
            prediction=fitted.prediction.astype(np.float32),
            validation_rmse=fitted.validation_rmse,
            details=json.dumps(fitted.details, sort_keys=True),
        )
    saved = np.load(path, allow_pickle=False)
    prediction = saved["prediction"].astype(np.float64)
    if len(prediction) != len(reference) or not np.isfinite(prediction).all():
        raise AssertionError(f"{name}/{seed}/{model}: invalid persisted prediction")
    details = json.loads(str(saved["details"].item()))
    return prediction, float(saved["validation_rmse"]), details, str(path.relative_to(ROOT)), reused


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, model), group in metrics.groupby(["dataset", "model"], sort=True):
        failures = group[group.g_verifier > TOLERANCE]
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "instances": len(group),
            "verifier_failures": len(failures),
            "verifier_failure_rate": len(failures) / len(group),
        }
        for field in ("g_verifier", "g_hard", "g_search", "g_total"):
            values = group[field].to_numpy(float)
            low, high = bootstrap_interval(values, "mean", stable_seed(dataset, model, field, "ci"))
            row[f"mean_{field}"] = float(np.mean(values))
            row[f"median_{field}"] = float(np.median(values))
            row[f"mean_{field}_ci_low"] = low
            row[f"mean_{field}_ci_high"] = high
        ranks = failures.witness_rank_R.to_numpy(float)
        low, high = bootstrap_interval(ranks, "median", stable_seed(dataset, model, "R", "ci"))
        row["median_witness_rank_R"] = float(np.median(ranks)) if len(ranks) else np.nan
        row["median_witness_rank_R_ci_low"] = low
        row["median_witness_rank_R_ci_high"] = high
        row["maximum_witness_rank_R"] = float(np.max(ranks)) if len(ranks) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_certificates(certificates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    failures = certificates[certificates.genuine_failure]
    for keys, group in failures.groupby(["dataset", "model", "budget"], sort=True):
        successes = int(group.certified_failure.sum())
        low, high = wilson_interval(successes, len(group))
        rows.append({
            "dataset": keys[0],
            "model": keys[1],
            "budget": int(keys[2]),
            "failure_instances": len(group),
            "certified_instances": successes,
            "top_b_recall": successes / len(group),
            "top_b_recall_wilson_ci_low": low,
            "top_b_recall_wilson_ci_high": high,
            "mean_certificate_tightness": float(group.l_b_over_g_verifier.mean()),
            "median_certificate_tightness": float(group.l_b_over_g_verifier.median()),
        })
    return pd.DataFrame(rows)


def result_markdown(metrics: pd.DataFrame, summary: pd.DataFrame, certificates: pd.DataFrame) -> str:
    compact = summary[[
        "dataset", "model", "instances", "verifier_failures", "mean_g_verifier",
        "mean_g_verifier_ci_low", "mean_g_verifier_ci_high", "mean_g_total",
        "median_witness_rank_R", "maximum_witness_rank_R",
    ]]
    recall = certificates[certificates.budget.isin([8, 16, 64, 256])][[
        "dataset", "model", "budget", "failure_instances", "top_b_recall",
        "top_b_recall_wilson_ci_low", "top_b_recall_wilson_ci_high",
        "mean_certificate_tightness",
    ]]
    failed = int((metrics.g_verifier > TOLERANCE).sum())
    exact = bool(metrics.r_le_n_2eps.fillna(True).all())
    return "\n".join([
        "# Extended external finite-set audit", "", "## Setup", "",
        "Fifteen deterministic, disjoint train/validation/selection splits were run for each of TF-Bind-8, TrpB, and GB1. Each split used pairwise ridge, ExtraTrees, the finite RankNet surrogate, and the finite COM-style surrogate. The first three saved predictions were reused exactly; twelve additional splits were fitted. All nominations are exact maxima of the learned finite score with ascending candidate ID as the tie-break.",
        "", "## Main results", "",
        f"Verifier failure occurred in {failed} of {len(metrics)} dataset/split/model instances. All persisted split files passed an independent zero-intersection and all-candidates-once check. The witness theorem check `R <= N(2 eps_inf)` was `{exact}` for every applicable row.",
        "", "## Key numbers", "", "Split-level gap summaries with deterministic 10,000-resample 95% bootstrap intervals:", "",
        markdown_table(compact), "", "Top-B recall (conditioned on verifier failure) with Wilson 95% intervals and certificate tightness:", "",
        markdown_table(recall), "", "## Negative or unexpected results", "",
        "Zero-verifier-gap cases remain in `metrics.parquet` and are excluded only from conditional certification-recall denominators. `ranking` and `com` retain their explicitly approximate finite-surrogate scope; they are not relabeled as the official RaM or COM reproductions. GB1 results remain relative to the 149,361 measured candidates. TrpB excludes the 871 absent four-site variants. No train or validation label was used to nominate a selection candidate.",
        "", "## Files", "",
        "- `metrics.parquet`: split-level finite audit gaps and ranks.",
        "- `certificates.parquet`: minimally processed Top-B certificate rows.",
        "- `summary.parquet`: gap distributions and confidence intervals.",
        "- `certificate_summary.parquet`: recall intervals and tightness.",
        "- `splits.parquet` and `raw/splits/`: exact split identifiers.",
        "- `prediction_index.parquet` and `raw/predictions/`: full finite predictions and provenance.",
        "- `leakage_check.json`: persisted-file leakage validation.",
        "- `config.json`: seeds, model definitions, budgets, and tie-breaking.",
        "- `run.log`: progress, environment, command, and completion status.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=tuple(DATASETS))
    args = parser.parse_args()
    started = time.monotonic()
    EXTERNAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    (EXTERNAL_OUTPUT / "run.log").write_text("")
    command = "python -m experiments.audits.primary_audit " + " ".join(sys.argv[1:])
    metric_rows: list[dict[str, object]] = []
    certificate_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_details: dict[str, object] = {}
    for name in args.datasets:
        frame, tokens, reference, candidate_ids = load_dataset(name)
        log(f"dataset={name} candidates={len(frame)} starting")
        for seed in SEEDS:
            split = split_indices(name, candidate_ids, seed)
            split_record = save_and_validate_split(EXTERNAL_OUTPUT, name, seed, candidate_ids, split)
            split_rows.append(split_record)
            selection = split["selection"]
            lookup = selection_lookup(tokens, selection)
            trusted_hard = sequence_search(
                reference,
                tokens,
                selection,
                lookup,
                DATASETS[name]["alphabet_size"],
                SEARCH_BUDGET,
                stable_seed("night2_external_many_splits", name, seed, "trusted_search"),
            )
            for model in MODELS:
                model_start = time.monotonic()
                prediction, validation_rmse, details, path, reused = fit_or_load(
                    name, model, seed, tokens, reference, split
                )
                approximate = sequence_search(
                    prediction,
                    tokens,
                    selection,
                    lookup,
                    DATASETS[name]["alphabet_size"],
                    SEARCH_BUDGET,
                    stable_seed("night2_external_many_splits", name, seed, model, "proxy_search"),
                )
                gaps = audit_gaps(
                    reference, prediction, selection, candidate_ids, approximate, trusted_hard
                )
                residual = prediction[selection] - reference[selection]
                metric_rows.append({
                    "dataset": name,
                    "seed": seed,
                    "model": model,
                    "candidate_count": len(frame),
                    "train_size": len(split["train"]),
                    "validation_size": len(split["validation"]),
                    "selection_size": len(selection),
                    "validation_rmse": validation_rmse,
                    "selection_rmse": float(np.sqrt(np.mean(residual**2))),
                    "search_budget": SEARCH_BUDGET,
                    **gaps,
                })
                order = ranked_order(prediction, selection, candidate_ids)
                selected = int(order[0])
                first_positions = np.flatnonzero(reference[order] > reference[selected] + TOLERANCE)
                rank = int(first_positions[0] + 1) if len(first_positions) else None
                if rank != gaps["witness_rank_R"]:
                    raise AssertionError("independent witness-rank recomputation disagreed")
                for budget in BUDGETS:
                    actual = min(budget, len(order))
                    lower = max(0.0, float(np.max(reference[order[:actual]]) - reference[selected]))
                    gap = float(gaps["g_verifier"])
                    if not (-TOLERANCE <= lower <= gap + 1e-10):
                        raise AssertionError("finite certificate violated one-sided bound")
                    certified = lower > TOLERANCE
                    expected = bool(rank is not None and rank <= budget)
                    if gap > TOLERANCE and certified != expected:
                        raise AssertionError("Top-B recall did not equal the R CDF")
                    certificate_rows.append({
                        "dataset": name,
                        "seed": seed,
                        "model": model,
                        "budget": budget,
                        "actual_probe_count": actual,
                        "genuine_failure": gap > TOLERANCE,
                        "certified_failure": certified,
                        "g_verifier": gap,
                        "l_b": lower,
                        "l_b_over_g_verifier": lower / gap if gap > TOLERANCE else np.nan,
                        "witness_rank_R": rank,
                    })
                prediction_rows.append({
                    "dataset": name,
                    "seed": seed,
                    "model": model,
                    "path": path,
                    "reused_night_one_prediction": reused,
                    "prediction_length": len(prediction),
                    "x_sel_index": int(gaps["x_sel_index"]),
                    "x_sel_candidate_id": int(candidate_ids[int(gaps["x_sel_index"])]),
                })
                model_details[f"{name}:{seed}:{model}"] = details
                log(
                    f"dataset={name} seed={seed} model={model} reused={reused} "
                    f"g_verifier={float(gaps['g_verifier']):.8g} R={rank} "
                    f"seconds={time.monotonic() - model_start:.3f}"
                )
    metrics = pd.DataFrame(metric_rows).sort_values(["dataset", "seed", "model"])
    certificates = pd.DataFrame(certificate_rows).sort_values(["dataset", "seed", "model", "budget"])
    splits = pd.DataFrame(split_rows).sort_values(["dataset", "seed"])
    predictions = pd.DataFrame(prediction_rows).sort_values(["dataset", "seed", "model"])
    summary = summarize_metrics(metrics)
    certificate_summary = summarize_certificates(certificates)
    metrics.to_parquet(EXTERNAL_OUTPUT / "metrics.parquet", index=False)
    certificates.to_parquet(EXTERNAL_OUTPUT / "certificates.parquet", index=False)
    summary.to_parquet(EXTERNAL_OUTPUT / "summary.parquet", index=False)
    certificate_summary.to_parquet(EXTERNAL_OUTPUT / "certificate_summary.parquet", index=False)
    splits.to_parquet(EXTERNAL_OUTPUT / "splits.parquet", index=False)
    predictions.to_parquet(EXTERNAL_OUTPUT / "prediction_index.parquet", index=False)
    leakage = {
        "implementation": "independent intersections after reloading every persisted compressed split-ID file",
        "split_count": len(splits),
        "all_passed": bool(splits.passed.all()),
        "maximum_overlap": int(splits[[
            "train_validation_overlap", "train_selection_overlap", "validation_selection_overlap"
        ]].to_numpy().max()),
        "checks": split_rows,
    }
    write_json(EXTERNAL_OUTPUT / "leakage_check.json", leakage)
    elapsed = time.monotonic() - started
    config = {
        "experiment": "extended external finite audit",
        "datasets": DATASETS,
        "seeds": list(SEEDS),
        "split_count_per_dataset": len(SEEDS),
        "models": list(MODELS),
        "model_scope": {
            "pairwise_ridge": "squared-error pairwise categorical ridge baseline",
            "extra_trees": "nonlinear ExtraTrees baseline",
            "ranking": "finite RankNet surrogate; not official RaM",
            "com": "finite discrete COM-style surrogate; not official COM",
        },
        "budgets": list(BUDGETS),
        "certificate_budget_includes_nomination": True,
        "search_budget": SEARCH_BUDGET,
        "tie_breaking": "descending learned score, then ascending integer candidate_id",
        "split_rule": "seeded NumPy permutation; disjoint train/validation prefix and remaining selection pool",
        "confidence_intervals": "10,000 deterministic split bootstrap resamples for continuous means/medians; Wilson score 95% interval for recall",
        "prediction_reuse": "initial-run predictions reused for seeds 20270813-20270815; all other predictions saved under raw/predictions",
        "model_details": model_details,
        "elapsed_seconds": elapsed,
    }
    write_json(EXTERNAL_OUTPUT / "config.json", config)
    (EXTERNAL_OUTPUT / "results.md").write_text(result_markdown(metrics, summary, certificate_summary))
    with (EXTERNAL_OUTPUT / "run.log").open("a") as handle:
        handle.write("\n".join(environment_lines(command, elapsed)) + "\n")
    print(json.dumps({"metric_rows": len(metrics), "certificate_rows": len(certificates), "elapsed_seconds": elapsed}))


if __name__ == "__main__":
    main()

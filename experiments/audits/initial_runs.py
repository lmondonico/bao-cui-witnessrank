"""Initial audits: three splits and five proxy families per landscape."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from experiments.audits.models import ModelFit, fit_model
from experiments.audits.policies import (
    TOLERANCE,
    audit_gaps,
    certificate_rows,
    disagreement_order,
    diversity_order,
    local_order,
    random_order,
    ranked_order,
    selection_lookup,
    sequence_search,
    top_order,
)
from experiments.paths import DATA as DATA_ROOT
from experiments.paths import AUDITS_INITIAL
from experiments.paths import ROOT


SEEDS = [20270813, 20270814, 20270815]
BUDGETS = [8, 16, 32, 64, 128, 256]
SEARCH_BUDGET = 256
DATASETS = {
    "tf_bind_8": {"alphabet_size": 4, "train": 4096, "validation": 2048},
    "trpb": {"alphabet_size": 20, "train": 4960, "validation": 1240},
    "gb1": {"alphabet_size": 20, "train": 4960, "validation": 620},
}
DATASET_OUTPUTS = {
    "tf_bind_8": AUDITS_INITIAL / "tf_bind_8",
    "trpb": AUDITS_INITIAL / "trpb",
    "gb1": AUDITS_INITIAL / "gb1",
}
MODELS = ["pairwise_ridge", "extra_trees", "ranking", "com"]


def stable_seed(*values: object) -> int:
    import hashlib

    digest = hashlib.blake2b("|".join(map(str, values)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_parquet(DATA_ROOT / name / "candidates.parquet")
    token_columns = sorted(
        (column for column in frame if column.startswith("token_")),
        key=lambda column: int(column.removeprefix("token_")),
    )
    tokens = frame[token_columns].to_numpy(np.int8)
    return (
        frame,
        tokens,
        frame.f.to_numpy(float),
        frame.candidate_id.to_numpy(np.int64),
    )


def split_indices(candidate_ids: np.ndarray, seed: int, train_size: int, validation_size: int) -> dict[str, np.ndarray]:
    order = np.random.default_rng(seed).permutation(len(candidate_ids))
    train = np.sort(order[:train_size])
    validation = np.sort(order[train_size:train_size + validation_size])
    selection = np.sort(order[train_size + validation_size:])
    result = {"train": train, "validation": validation, "selection": selection}
    sets = {name: set(candidate_ids[indices].tolist()) for name, indices in result.items()}
    overlaps = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_selection": len(sets["train"] & sets["selection"]),
        "validation_selection": len(sets["validation"] & sets["selection"]),
    }
    if any(overlaps.values()):
        raise AssertionError(f"data leakage detected: {overlaps}")
    result["overlaps"] = overlaps  # type: ignore[assignment]
    return result


def split_frame(candidate_ids: np.ndarray, split: dict[str, np.ndarray], seed: int) -> pd.DataFrame:
    frames = []
    for name in ("train", "validation", "selection"):
        indices = split[name]
        frames.append(pd.DataFrame({
            "seed": seed,
            "split": name,
            "candidate_index": indices,
            "candidate_id": candidate_ids[indices],
        }))
    return pd.concat(frames, ignore_index=True)


def _prediction_path(dataset: str, seed: int, model: str) -> Path:
    directory = DATASET_OUTPUTS[dataset] / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"seed_{seed}_{model}.npz"


def run_dataset(name: str, models: list[str], force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.monotonic()
    frame, tokens, reference, candidate_ids = load_dataset(name)
    specification = DATASETS[name]
    alphabet_size = int(specification["alphabet_size"])
    metric_rows: list[dict[str, object]] = []
    certificate_data: list[dict[str, object]] = []
    prediction_index: list[dict[str, object]] = []
    split_frames: list[pd.DataFrame] = []
    model_details: dict[str, object] = {}
    for seed in SEEDS:
        split = split_indices(
            candidate_ids, seed, int(specification["train"]), int(specification["validation"])
        )
        split_frames.append(split_frame(candidate_ids, split, seed))
        selection = split["selection"]
        leakage = split["overlaps"]
        lookup = selection_lookup(tokens, selection)
        trusted_hard = sequence_search(
            reference, tokens, selection, lookup, alphabet_size, SEARCH_BUDGET,
            stable_seed(name, seed, "trusted"),
        )
        fits: dict[str, ModelFit] = {}
        for model_name in models:
            prediction_path = _prediction_path(name, seed, model_name)
            if prediction_path.exists() and not force:
                saved = np.load(prediction_path, allow_pickle=False)
                prediction = saved["prediction"].astype(float)
                details = json.loads(str(saved["details"].item()))
                fit = ModelFit(prediction, float(saved["validation_rmse"]), details)
            else:
                model_seed = stable_seed(name, seed, model_name)
                fit = fit_model(
                    model_name, tokens, reference, split["train"], split["validation"],
                    selection, alphabet_size, model_seed,
                )
                np.savez_compressed(
                    prediction_path,
                    prediction=fit.prediction.astype(np.float32),
                    validation_rmse=fit.validation_rmse,
                    details=json.dumps(fit.details, sort_keys=True),
                )
            fits[model_name] = fit
            model_details[f"{seed}:{model_name}"] = fit.details
        for model_name, fit in fits.items():
            prediction = fit.prediction
            proxy_order = ranked_order(prediction, selection, candidate_ids)
            selected = int(proxy_order[0])
            approximate = sequence_search(
                prediction, tokens, selection, lookup, alphabet_size, SEARCH_BUDGET,
                stable_seed(name, seed, model_name, "proxy_search"),
            )
            gaps = audit_gaps(reference, prediction, selection, candidate_ids, approximate, trusted_hard)
            if gaps["gap_identity_error"] > 1e-9:
                raise AssertionError("trusted gap identity failed")
            selection_residual = prediction[selection] - reference[selection]
            metric_rows.append({
                "dataset": name,
                "seed": seed,
                "model": model_name,
                "candidate_count": len(frame),
                "train_size": len(split["train"]),
                "validation_size": len(split["validation"]),
                "selection_size": len(selection),
                "train_validation_overlap": leakage["train_validation"],
                "train_selection_overlap": leakage["train_selection"],
                "validation_selection_overlap": leakage["validation_selection"],
                "validation_rmse": fit.validation_rmse,
                "selection_rmse": float(np.sqrt(np.mean(selection_residual**2))),
                "search_budget": SEARCH_BUDGET,
                **gaps,
            })
            prediction_index.append({
                "dataset": name,
                "seed": seed,
                "model": model_name,
                "path": str(_prediction_path(name, seed, model_name).relative_to(ROOT)),
                "x_sel_index": selected,
                "x_sel_candidate_id": int(candidate_ids[selected]),
                "x_sel_sequence": str(frame.sequence.iloc[selected]),
            })
            other_name = next((candidate for candidate in ("pairwise_ridge", "extra_trees", "mlp")
                               if candidate != model_name and candidate in fits), None)
            orders = {
                "top_b": top_order(selected, prediction, selection, candidate_ids),
                "local_neighborhood": local_order(selected, prediction, tokens, selection, candidate_ids),
                "diversity_high_ranked": diversity_order(
                    selected, prediction, tokens, selection, candidate_ids, max(BUDGETS)
                ),
            }
            if other_name is not None:
                orders["ensemble_disagreement"] = disagreement_order(
                    selected, prediction, fits[other_name].prediction, selection, candidate_ids
                )
            for policy, order in orders.items():
                certificate_data.extend(certificate_rows(
                    name, model_name, seed, reference, selected, order, policy, BUDGETS,
                    float(gaps["g_verifier"]), gaps["witness_rank_R"],
                ))
            for repetition in range(10):
                order = random_order(selected, selection, stable_seed(name, seed, model_name, "random", repetition))
                certificate_data.extend(certificate_rows(
                    name, model_name, seed, reference, selected, order, "random", BUDGETS,
                    float(gaps["g_verifier"]), gaps["witness_rank_R"], repetition,
                ))
    config = {
        "dataset": name,
        "dataset_audit": json.loads((DATA_ROOT / name / "dataset_audit.json").read_text()),
        "seeds": SEEDS,
        "models": models,
        "budgets": BUDGETS,
        "random_probe_repeats": 10,
        "search_budget": SEARCH_BUDGET,
        "split_sizes": {
            "train": int(specification["train"]),
            "validation": int(specification["validation"]),
            "selection": len(frame) - int(specification["train"]) - int(specification["validation"]),
        },
        "split_rule": "seeded permutation, sorted indices within each disjoint split",
        "ranking_and_tie_breaking": "descending proxy score, then ascending integer candidate_id; candidate_id is lexical base-alphabet encoding",
        "certificate_budget_includes_nomination": True,
        "probe_policy_details": {
            "top_b": "descending proxy score, ascending candidate_id tie-break",
            "local_neighborhood": "ascending Hamming distance from nomination, then descending proxy score, then candidate_id",
            "random": "nomination first, then seeded permutation; 10 repetitions per instance",
            "ensemble_disagreement": "descending absolute disagreement from first available different baseline in priority pairwise_ridge, extra_trees, mlp; then proxy score and candidate_id",
            "diversity_high_ranked": "greedy max-min Hamming distance from selected probes over the top max(4096,16*Bmax) proxy candidates; proxy score and candidate_id tie-break",
        },
        "torch_deterministic_algorithms": True,
        "model_details": model_details,
        "elapsed_seconds": time.monotonic() - started,
    }
    return (
        pd.DataFrame(metric_rows), pd.DataFrame(certificate_data),
        pd.concat(split_frames, ignore_index=True), config | {"prediction_index": prediction_index},
    )


def summarize_certificates(certificates: pd.DataFrame) -> pd.DataFrame:
    genuine = certificates[certificates.genuine_failure]
    if genuine.empty:
        return pd.DataFrame(columns=[
            "dataset", "model", "policy", "budget", "instances", "certification_recall",
            "mean_l_b_over_g_verifier", "median_trusted_evaluations_to_first_positive",
        ])
    return genuine.groupby(["dataset", "model", "policy", "budget"], as_index=False).agg(
        instances=("g_verifier", "size"),
        certification_recall=("certified_failure", "mean"),
        mean_l_b_over_g_verifier=("l_b_over_g_verifier", "mean"),
        median_trusted_evaluations_to_first_positive=("trusted_evaluations_to_first_positive", "median"),
    )


def top_b_exact_check(certificates: pd.DataFrame) -> dict[str, object]:
    values = certificates[
        (certificates.policy == "top_b") & certificates.genuine_failure
    ].top_b_recall_equals_r_cdf.dropna()
    return {"applicable_rows": int(len(values)), "all_exact": bool(values.astype(bool).all())}


def independent_leakage_check(splits: pd.DataFrame) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    for seed, group in splits.groupby("seed"):
        values = {
            name: set(group.loc[group.split == name, "candidate_id"].astype(int).tolist())
            for name in ("train", "validation", "selection")
        }
        check: dict[str, object] = {
            "seed": int(seed),
            "train_validation_overlap": len(values["train"] & values["validation"]),
            "train_selection_overlap": len(values["train"] & values["selection"]),
            "validation_selection_overlap": len(values["validation"] & values["selection"]),
            "total_unique_candidate_ids": len(set.union(*values.values())),
            "total_rows": len(group),
        }
        overlap_keys = ("train_validation_overlap", "train_selection_overlap", "validation_selection_overlap")
        check["passed"] = not any(int(check[key]) for key in overlap_keys) and (
            check["total_unique_candidate_ids"] == check["total_rows"]
        )
        checks.append(check)
    return {
        "implementation": "independent set-intersection pass over persisted split rows",
        "checks": checks,
        "all_passed": all(bool(check["passed"]) for check in checks),
    }


def markdown_table(frame: pd.DataFrame, include_index: bool = True) -> str:
    display = frame.reset_index() if include_index else frame.copy()
    columns = [str(column) for column in display.columns]
    rows = []
    for values in display.itertuples(index=False, name=None):
        rows.append([
            f"{value:.3f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value)
            for value in values
        ])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_log(directory: Path, command: str, config: dict[str, object], error: str | None = None) -> None:
    text = [
        f"started_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command}",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scipy={scipy.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"torch={torch.__version__}",
        "pyarrow=21.0.0",
        f"torch_deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
        f"elapsed_seconds={config.get('elapsed_seconds')}",
    ]
    if error:
        text.append("error_traceback:\n" + error)
    else:
        text.append("status=completed")
    (directory / "run.log").write_text("\n".join(text) + "\n")


def results_markdown(name: str, metrics: pd.DataFrame, summary: pd.DataFrame, config: dict[str, object]) -> str:
    failures = metrics[metrics.g_verifier > TOLERANCE]
    top = summary[summary.policy == "top_b"]
    lines = [
        f"# {name}", "", "## Setup", "",
        f"Three disjoint seeded splits were audited over {int(metrics.candidate_count.iloc[0]):,} finite candidates. "
        f"Training/validation/selection sizes were {config['split_sizes']['train']:,}/"
        f"{config['split_sizes']['validation']:,}/{config['split_sizes']['selection']:,}. "
        f"Models: {', '.join(config['models'])}.", "", "## Main results", "",
        f"Verifier failure occurred in {len(failures)} of {len(metrics)} dataset/seed/model instances. "
        f"All three independently checked split intersections were zero for every instance. "
        f"The Top-B witness-rank identity check was `{config['top_b_exact_check']['all_exact']}`.",
        "", "## Key numbers", "",
    ]
    if len(top):
        audit_table = metrics.groupby("model", as_index=False).agg(
            mean_g_verifier=("g_verifier", "mean"),
            mean_g_total=("g_total", "mean"),
            mean_selection_rmse=("selection_rmse", "mean"),
            median_witness_rank_R=("witness_rank_R", "median"),
            maximum_witness_rank_R=("witness_rank_R", "max"),
        )
        lines.extend(["Finite audit metrics:", "", markdown_table(audit_table, include_index=False), ""])
        table = top.pivot(index="model", columns="budget", values="certification_recall")
        lines.extend(["Top-B certification recall among genuine verifier failures:", "", markdown_table(table), ""])
    limitations = (
        "TrpB uses 159,129 directly quantified non-stop variants; 871 missing variants remain absent and the "
        "separately released KNN-imputed table is not used."
        if config["dataset"] == "trpb" else
        "TF-Bind-8 contains 256 duplicate source rows, but each duplicate pair has identical labels; the audit "
        "deduplicates with the same last-occurrence convention as the exact oracle."
        if config["dataset"] == "tf_bind_8" else
        "GB1 claims are limited to 149,361 measured candidates and do not extend to unmeasured variants."
    )
    lines.extend([
        "## Negative or unexpected results", "",
        "Instances with no verifier failure are retained in `metrics.parquet` and excluded only from recall denominators. "
        "The ranking result is an implemented RankNet surrogate, not a reproduction of a broader offline-MBO optimizer. "
        "The COM result is an explicitly documented finite discrete COM-style adaptation. " + limitations, "",
        "## Files", "",
        "- `metrics.parquet`: one row per dataset/seed/model audit.",
        "- `certificates.parquet`: minimally processed per-budget probe results.",
        "- `certificate_summary.parquet`: policy summaries over genuine verifier failures.",
        "- `splits.parquet`: exact candidate IDs for all disjoint splits.",
        "- `leakage_check.json`: independent intersection check over persisted split identifiers.",
        "- `predictions/`: full finite-set predictions and fitted-model metadata.",
        "- `config.json`: exact seeds, budgets, split rules, objectives, and tie-breaking.",
        "- `run.log`: command, versions, duration, and completion status.", "",
    ])
    return "\n".join(lines)


def write_dataset_results(name: str, metrics: pd.DataFrame, certificates: pd.DataFrame, splits: pd.DataFrame, config: dict[str, object], command: str) -> None:
    result_name = DATASET_OUTPUTS[name].name
    directory = DATASET_OUTPUTS[name]
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize_certificates(certificates)
    metrics.to_parquet(directory / "metrics.parquet", index=False)
    certificates.to_parquet(directory / "certificates.parquet", index=False)
    summary.to_parquet(directory / "certificate_summary.parquet", index=False)
    splits.to_parquet(directory / "splits.parquet", index=False)
    config["top_b_exact_check"] = top_b_exact_check(certificates)
    leakage_check = independent_leakage_check(pd.read_parquet(directory / "splits.parquet"))
    if not leakage_check["all_passed"]:
        raise AssertionError("independent persisted-split leakage check failed")
    config["independent_leakage_check"] = leakage_check
    (directory / "leakage_check.json").write_text(json.dumps(leakage_check, indent=2) + "\n")
    (directory / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (directory / "results.md").write_text(results_markdown(result_name, metrics, summary, config))
    write_log(directory, command, config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    command = "python -m experiments.audits.initial_runs " + " ".join(sys.argv[1:])
    all_metrics: list[pd.DataFrame] = []
    for name in args.datasets:
        print(f"[{name}] starting", flush=True)
        directory = DATASET_OUTPUTS[name]
        directory.mkdir(parents=True, exist_ok=True)
        try:
            metrics, certificates, splits, config = run_dataset(name, args.models, args.force)
            write_dataset_results(name, metrics, certificates, splits, config, command)
            failure_path = directory / "failure.md"
            if failure_path.exists():
                failure_path.unlink()
            all_metrics.append(metrics)
            print(f"[{name}] completed {len(metrics)} audits, {len(certificates)} certificate rows", flush=True)
        except Exception:
            error = traceback.format_exc()
            write_log(directory, command, {}, error)
            (directory / "failure.md").write_text(f"# Failed run\n\n```text\n{error}\n```\n")
            print(error, file=sys.stderr, flush=True)
    if not all_metrics or len(all_metrics) != len(args.datasets):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

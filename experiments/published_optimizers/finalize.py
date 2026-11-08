"""Check and summarize the published-optimizer runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paths import OPTIMIZERS_COM, OPTIMIZERS_RAM
from experiments.published_optimizers.common import TOLERANCE


def validate_splits(output: Path, expected: int) -> pd.DataFrame:
    rows = []
    for path in sorted((output / "raw" / "splits").glob("*.npz")):
        saved = np.load(path, allow_pickle=False)
        ids = {name: saved[f"{name}_candidate_ids"] for name in ("train", "validation", "selection")}
        overlaps = {
            "train_validation_overlap": len(set(ids["train"]) & set(ids["validation"])),
            "train_selection_overlap": len(set(ids["train"]) & set(ids["selection"])),
            "validation_selection_overlap": len(set(ids["validation"]) & set(ids["selection"])),
        }
        combined = np.concatenate(tuple(ids.values()))
        rows.append({
            "split_file": str(path.relative_to(output)),
            "train_size": len(ids["train"]),
            "validation_size": len(ids["validation"]),
            "selection_size": len(ids["selection"]),
            "total_size": len(combined),
            **overlaps,
            "passed": not any(overlaps.values()) and len(np.unique(combined)) == len(combined),
        })
    frame = pd.DataFrame(rows)
    if len(frame) != expected or not frame.passed.all():
        raise AssertionError(f"split validation failed: {len(frame)}/{expected}")
    frame.to_parquet(output / "leakage_checks.parquet", index=False)
    return frame


def make_results(method: str, output: Path, metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    title = "Offline-RaM full-scale component integration" if method == "ram" else "COM full-scale component integration"
    complete = metrics[metrics.status == "complete"]
    recall = complete.groupby(["dataset", "nomination_type", "budget"], sort=True).certified.mean()
    delivery = pd.read_parquet(output / "method_delivery_audit.parquet")
    lines = [
        f"# {title}", "", "## Setup", "",
        "Pinned official author components were run on TF-Bind-8, TrpB, and GB1 over five fixed splits. "
        "Exact learned-score and decoded method-delivered nominations are audited separately. Invalid decoded candidates "
        "are retained in raw proposal tables and never projected.", "",
        (
            "This is a finite-landscape official-component adaptation at all published TF-Bind-8 method hyperparameters. "
            "It is not a native Design-Bench reproduction; exact native failures and compatibility changes are recorded "
            "in `sanity_checks.json` and `config.json`."
            if method == "com" else
            "This is a finite-landscape official-component adaptation matching published width, epochs, list length, and "
            "search scale, with a smaller fixed list corpus than the published schedule. It is not a native Design-Bench reproduction."
        ), "",
        "## Main results", "",
        "Certification recall by dataset, nomination, and budget:", "",
        "| Dataset | Nomination | " + " | ".join(f"B={b}" for b in sorted(metrics.budget.unique())) + " |",
        "|---|---|" + "---|" * metrics.budget.nunique(),
    ]
    for dataset in sorted(metrics.dataset.unique()):
        for nomination in sorted(metrics.nomination_type.unique()):
            key = (dataset, nomination)
            if key in recall.index.droplevel("budget"):
                values = recall.loc[key]
                cells = [f"{values.loc[b]:.3f}" for b in sorted(metrics.budget.unique())]
            else:
                cells = ["NA"] * metrics.budget.nunique()
            lines.append(f"| {dataset} | {nomination} | " + " | ".join(cells) + " |")
    lines += ["", "True method delivery eligibility by dataset:", "", "| Dataset | Eligible runs | Ineligible runs |", "|---|---:|---:|"]
    for dataset in sorted(delivery.dataset.unique()):
        part = delivery[delivery.dataset == dataset]
        good = int(part.true_delivery_selection_eligible.sum())
        lines.append(f"| {dataset} | {good} | {len(part) - good} |")
    lines += ["", "## Key numbers", ""]
    for dataset in sorted(metrics.dataset.unique()):
        subset = metrics[(metrics.dataset == dataset) & (metrics.budget == metrics.budget.max())]
        for nomination in sorted(subset.nomination_type.unique()):
            current = subset[(subset.nomination_type == nomination) & (subset.status == "complete")]
            if current.empty:
                lines.append(f"- {dataset}, {nomination}: no eligible nomination; metrics unavailable.")
                continue
            lines.append(
                f"- {dataset}, {nomination}: G_verifier median {current.g_verifier.median():.6g}, "
                f"range [{current.g_verifier.min():.6g}, {current.g_verifier.max():.6g}]; "
                f"median witness rank {current.witness_rank.median():.1f}."
            )
    outside = int(diagnostics.ineligible_proposal_count.sum())
    total = int(diagnostics.proposal_count.sum())
    eligible = int(diagnostics.eligible_unique_proposal_count.sum())
    lines += ["", "## Negative or unexpected results", "",
              f"- {outside} of {total} decoded proposals were outside selection pools. Across runs, {eligible} unique eligible proposals were retained.",
              f"- The true highest-score decoded method delivery was eligible in {int(delivery.true_delivery_selection_eligible.sum())} of {len(delivery)} runs. Ineligible deliveries have unavailable finite-selection G/L_B/R fields and are not replaced by the best eligible candidate."]
    if method == "ram":
        benchmark = json.loads((output / "raw" / "scale_benchmark.json").read_text())
        lines.append("- The native Design-Bench attempt remains incompatible with the current environment/data distribution; the exact traceback is retained. The official theory example also produces NaN RankCosine parameters.")
        lines.append("- Although width, epochs, list length, and optimizer search match defaults, online one-list updates do not reproduce the official fixed 10,000-list, batch-128 training corpus.")
        lines.append(
            f"- A literal batch benchmark took {benchmark['seconds_per_literal_batch_update']:.3f} seconds and "
            f"{benchmark['peak_cuda_bytes'] / 2**30:.2f} GiB. Linear training-only projection was "
            f"{benchmark['projected_hours_100_epochs']:.2f} GPU-hours per split and "
            f"{benchmark['projected_gpu_hours_15_runs']:.2f} GPU-hours for 15 runs, excluding validation, "
            "fixed-corpus construction, search, and audit."
        )
    else:
        lines.append("- The native CLI and TF2.4-era aggregation wrapper outcomes are retained. The finite adapter uses unchanged per-batch train_step/validate_step under TF2.20.")
        lines.append("- TF-Bind-8 script hyperparameters were transferred to TrpB and GB1 without dataset-specific tuning.")
        histories = pd.concat([
            pd.read_parquet(path) for path in sorted((output / "raw" / "training").glob("*.parquet"))
        ], ignore_index=True)
        final = histories.sort_values("epoch").groupby(["dataset", "seed"], as_index=False).last()
        over_min = final["train:train/overestimation"].min()
        over_max = final["train:train/overestimation"].max()
        alpha_min = final["train:train/alpha"].min()
        alpha_max = final["train:train/alpha"].max()
        lines.append(
            f"- Final-epoch mean adversarial overestimation ranged from {over_min:.6g} to {over_max:.6g}; "
            f"learned alpha ranged from {alpha_min:.6g} to {alpha_max:.6g}. These extreme negative values and "
            "near-zero alphas are preserved, but this run cannot determine whether they reflect intended COM dynamics "
            "under these landscapes or the TF2.4-to-TF2.20 integration."
        )
    lines += ["", "## Files", "", "- `metrics.parquet`: budget-level audit metrics.",
              "- `diagnostics.parquet`: proposal eligibility and nominations.",
              "- `method_delivery_audit.parquet`: true decoded method delivery, score, split, and eligibility per run.",
              "- `raw/`: splits, predictions, proposals, histories, and checkpoints.",
              "- `config.json`, `provenance.json`, `sanity_checks.json`, `run.log`: reproducibility metadata.", ""]
    (output / "results.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("ram", "com"))
    args = parser.parse_args()
    output = {
        "ram": OPTIMIZERS_RAM,
        "com": OPTIMIZERS_COM,
    }[args.method]
    metrics = pd.read_parquet(output / "metrics.parquet")
    diagnostics = pd.read_parquet(output / "diagnostics.parquet")
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    expected_runs = len(config["datasets"]) * len(config["seeds"])
    expected_rows = expected_runs * 3 * len(config["budgets"])
    if len(metrics) != expected_rows or len(diagnostics) != expected_runs:
        raise AssertionError("incomplete budget or run rows")
    if not set(metrics.status).issubset({"complete", "ineligible_method_delivery"}) or config["failed_runs"]:
        raise AssertionError("technical production failure present")
    complete = metrics[metrics.status == "complete"]
    ineligible = metrics[metrics.status == "ineligible_method_delivery"]
    if not ((complete.l_b >= -TOLERANCE) & (complete.l_b <= complete.g_verifier + 1e-9)).all():
        raise AssertionError("certificate bound failure")
    implied = complete.witness_rank.notna() & (complete.witness_rank <= complete.budget)
    if not np.array_equal(implied.to_numpy(), complete.certified.astype(bool).to_numpy()):
        raise AssertionError("witness-rank recall identity failure")
    unavailable = ["g_verifier", "g_total", "g_search_proxy", "witness_rank", "l_b", "l_b_over_g_verifier"]
    if not ineligible[unavailable].isna().all().all():
        raise AssertionError("ineligible true delivery has finite audit metrics")
    delivery = pd.read_parquet(output / "method_delivery_audit.parquet")
    if len(delivery) != expected_runs:
        raise AssertionError("missing true delivery audit")
    for row in delivery.itertuples(index=False):
        proposal = pd.read_parquet(
            output / "raw" / "proposals" / f"{row.dataset}_seed_{row.seed}.parquet"
        )
        winners = proposal[proposal.is_true_method_delivery]
        if len(winners) != 1 or int(winners.iloc[0].proposal_number) != row.true_delivery_proposal_number:
            raise AssertionError("true delivery marker mismatch")
        order = np.lexsort((proposal.proposal_number.to_numpy(), -proposal.learned_score.to_numpy()))
        if int(proposal.iloc[int(order[0])].proposal_number) != row.true_delivery_proposal_number:
            raise AssertionError("true delivery is not highest learned-score proposal")
        if bool(row.true_delivery_selection_eligible) != (row.true_delivery_split == "selection"):
            raise AssertionError("delivery eligibility mismatch")
    leakage = validate_splits(output, expected_runs)
    diagnostics["eligible_membership_fraction"] = (
        diagnostics.proposal_count - diagnostics.ineligible_proposal_count
    ) / diagnostics.proposal_count
    diagnostics["eligible_unique_fraction"] = (
        diagnostics.eligible_unique_proposal_count / diagnostics.proposal_count
    )
    diagnostics.to_parquet(output / "diagnostics.parquet", index=False)
    fractions = diagnostics.set_index(["dataset", "seed"])[
        ["eligible_membership_fraction", "eligible_unique_fraction"]
    ]
    metrics = metrics.drop(
        columns=["eligible_membership_fraction", "eligible_unique_fraction"], errors="ignore"
    )
    metrics = metrics.join(fractions, on=["dataset", "seed"])
    if args.method == "com":
        from experiments.published_optimizers.com import run as com_runner

        sanity_path = output / "sanity_checks.json"
        sanity = json.loads(sanity_path.read_text())
        previous = sanity[0]
        previous["name"] = "invalid_python_module_entry_attempt_retained"
        previous["passed"] = False
        previous["reason_invalid"] = (
            "design_baselines.coms_cleaned has no __main__; this was not the setuptools coms entry point"
        )
        corrected = com_runner.native_sanity()
        remainder = [
            item for item in sanity[1:]
            if item["name"] not in {
                "invalid_python_module_entry_attempt_retained",
                "official_native_tfbind8_published_cli_isolated",
            }
        ]
        sanity = [previous, corrected] + remainder
        sanity_path.write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
        config["corrected_native_cli_sanity"] = corrected["name"]
        config["exact_divergences_from_published_tf_bind_script"] = [
            "StaticGraphTask is replaced by the local finite-table loader and persisted candidate-disjoint splits",
            "the published 500-row validation split is replaced by the declared per-landscape validation pools",
            "TensorFlow 2.20/Keras 3 replaces documented TensorFlow 2.4, requiring the MSE alias and logging-wrapper bypass",
            "TF-Bind-8 hyperparameters are transferred to TrpB and GB1 without an author-provided task script",
        ]
    else:
        sanity_path = output / "sanity_checks.json"
        sanity = json.loads(sanity_path.read_text())
        raw = output / "raw"
        native_sharded = {
            "name": "official_native_sharded_direct_attempt",
            "passed": (raw / "native_sharded_attempt.returncode").read_text().strip() == "0",
            "returncode": int((raw / "native_sharded_attempt.returncode").read_text()),
            "stdout_artifact": "raw/native_sharded_attempt.stdout.log",
            "stderr_artifact": "raw/native_sharded_attempt.stderr.log",
            "failure": "official config enables W&B but argparse/default config supplies no wandb_api field",
        }
        native_wrapper = {
            "name": "official_native_sharded_no_wandb_wrapper_attempt",
            "passed": (raw / "native_sharded_wrapper.returncode").read_text().strip() == "0",
            "returncode": int((raw / "native_sharded_wrapper.returncode").read_text()),
            "stdout_artifact": "raw/native_sharded_wrapper.stdout.log",
            "stderr_artifact": "raw/native_sharded_wrapper.stderr.log",
            "compatibility_change": "wrapper sets only use_wandb=False before executing unchanged official main",
            "failure": "reduced 128-list sanity creates a 25-list validation split smaller than batch 32 with drop_last=True; official loop divides by zero-length validation loader",
        }
        sanity = [
            item for item in sanity
            if item["name"] not in {native_sharded["name"], native_wrapper["name"]}
        ]
        sanity.extend([native_sharded, native_wrapper])
        sanity_path.write_text(json.dumps(sanity, indent=2, sort_keys=True) + "\n")
        config["native_sharded_attempts"] = [native_sharded["name"], native_wrapper["name"]]
        benchmark_path = output / "raw" / "scale_benchmark.json"
        benchmark = json.loads(benchmark_path.read_text())
        if not benchmark.get("passed"):
            raise AssertionError("RAM published-batch scale benchmark failed")
        config["published_batch_scale_benchmark"] = {
            **benchmark,
            "artifact": str(benchmark_path.relative_to(output)),
        }
        config["list_scale_accounting"] = {
            "adapted_train_list_presentations": config["epochs"] * config["lists_per_epoch"],
            "published_train_list_presentations": 100 * 8000,
            "published_validation_list_presentations": 100 * 2000,
            "adapted_to_published_train_fraction": (
                config["epochs"] * config["lists_per_epoch"] / (100 * 8000)
            ),
            "note": "Published counts reflect the official 80/20 split of 10,000 fixed lists; adapter validation uses a separate declared candidate pool.",
        }
    config["production_command"] = {
        "script": f"experiments/published_optimizers/{args.method}/run.py",
        "arguments": {key: config[key] for key in (
            ("datasets", "seeds", "epochs", "list_length", "lists_per_epoch", "hidden_size", "search_solutions", "search_steps")
            if args.method == "ram"
            else ("datasets", "seeds", "epochs", "hidden_size", "batch_size", "particle_train_steps", "particle_evaluate_steps", "particle_lr", "overestimation_limit", "evaluation_samples")
        )},
        "cuda_visible_devices": "0" if args.method == "ram" else "1",
    }
    provenance_path = output / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    external = "ram" if args.method == "ram" else "com"
    provenance["night3_compatibility_environment"] = f".external/{external}/python_legacy"
    provenance["design_bench_install_log"] = f".external/{external}/install_design_bench.log"
    provenance["compatibility_package_inventory"] = f".external/{external}/python_legacy_packages.txt"
    provenance["numpy2_runtime_alias_patch"] = f".external/{external}/python_legacy/sitecustomize.py"
    provenance["tf8_native_shard_mapping"] = f".external/{external}/native_tf8_shard_mapping.json"
    provenance["native_dependency_install_logs"] = [
        f".external/{external}/install_design_bench.log",
        f".external/{external}/install_native_gym.log",
        f".external/{external}/install_native_morphing_agents.log",
        f".external/{external}/install_native_robel.log",
        f".external/{external}/install_native_transforms3d.log",
    ]
    if args.method == "ram":
        provenance["wandb_install_log"] = ".external/ram/install_wandb.log"
        provenance["literal_scale_benchmark"] = "raw/scale_benchmark.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    metrics["verifier_failure"] = metrics.g_verifier > TOLERANCE
    metrics["g_total"] = metrics.g_verifier
    metrics.to_parquet(output / "metrics.parquet", index=False)
    metrics.to_parquet(output / "raw_results.parquet", index=False)
    config["independent_validation"] = {
        "expected_runs": expected_runs,
        "observed_runs": len(diagnostics),
        "expected_budget_rows": expected_rows,
        "observed_budget_rows": len(metrics),
        "leakage_checks_passed": int(leakage.passed.sum()),
        "certificate_bounds_passed": True,
        "witness_rank_identity_passed": True,
        "true_delivery_highest_score_identity_passed": True,
        "ineligible_delivery_metrics_unavailable": True,
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    make_results(args.method, output, metrics, diagnostics)
    with (output / "run.log").open("a") as handle:
        handle.write(json.dumps({"event": "independent_validation", **config["independent_validation"]}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

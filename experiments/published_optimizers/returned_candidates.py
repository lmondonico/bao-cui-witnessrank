"""Separate each optimizer's returned candidate from its best in-pool proposal."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paths import OPTIMIZERS_COM, OPTIMIZERS_RAM, ROOT
from experiments.published_optimizers.common import (
    BUDGETS,
    DATASETS,
    TOLERANCE,
    design_bench_logits,
    deterministic_order,
    load_dataset,
    normalize_from_train,
)


def load_split(path: Path) -> dict[str, np.ndarray]:
    saved = np.load(path, allow_pickle=False)
    return {name: saved[f"{name}_indices"] for name in ("train", "validation", "selection")}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ram_scores(output: Path, dataset: str, seed: int, tokens: np.ndarray, split, config) -> np.ndarray:
    import torch

    ram_root = ROOT / ".external" / "ram"
    model_module = load_module("night3_delivery_ram_model", ram_root / "model.py")
    alphabet = DATASETS[dataset]["alphabet_size"]
    all_flat = design_bench_logits(tokens, alphabet).reshape(len(tokens), -1)
    _, mean, scale = normalize_from_train(all_flat, split["train"])
    proposal = np.load(
        output / "raw" / "predictions" / f"{dataset}_seed_{seed}.npz", allow_pickle=False
    )["proposal_tokens"]
    proposal_flat = design_bench_logits(proposal, alphabet).reshape(len(proposal), -1)
    features = ((proposal_flat - mean) / scale).astype(np.float32)
    model = model_module.SimpleMLP(features.shape[1], [config["hidden_size"]] * 2, 1)
    state = torch.load(
        output / "raw" / "checkpoints" / f"{dataset}_seed_{seed}.pt",
        map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        standardized = model(torch.from_numpy(features)).squeeze(-1).numpy()
    _, _, reference, _ = load_dataset(dataset)
    target_mean = float(reference[split["train"]].mean())
    target_scale = float(reference[split["train"]].std())
    return target_mean + target_scale * standardized


def com_scores(output: Path, dataset: str, seed: int, tokens: np.ndarray, split, config) -> np.ndarray:
    import tensorflow as tf

    com_root = ROOT / ".external" / "com"
    if not hasattr(tf.keras.losses, "mean_squared_error"):
        tf.keras.losses.mean_squared_error = lambda y, p: tf.reduce_mean(tf.square(y - p), axis=-1)
    nets = load_module(
        "night3_delivery_com_nets", com_root / "design_baselines" / "coms_cleaned" / "nets.py"
    )
    alphabet = DATASETS[dataset]["alphabet_size"]
    proposal = np.load(
        output / "raw" / "predictions" / f"{dataset}_seed_{seed}.npz", allow_pickle=False
    )["proposal_tokens"]
    features = design_bench_logits(proposal, alphabet)
    model = nets.ForwardModel(features.shape[1:], activations=("relu", "relu"), hidden_size=config["hidden_size"], final_tanh=False)
    model(features[:1], training=False)
    model.load_weights(output / "raw" / "checkpoints" / f"{dataset}_seed_{seed}.weights.h5")
    standardized = model(features, training=False).numpy()[:, 0]
    _, _, reference, _ = load_dataset(dataset)
    target_mean = float(reference[split["train"]].mean())
    target_scale = float(reference[split["train"]].std())
    return target_mean + target_scale * standardized


def audit_nomination(
    template: pd.Series,
    nomination_type: str,
    nomination: int | None,
    status: str,
    prediction: np.ndarray,
    reference: np.ndarray,
    candidate_ids: np.ndarray,
    selection: np.ndarray,
    delivery: pd.Series,
) -> list[dict[str, object]]:
    proxy_order = deterministic_order(prediction, selection, candidate_ids)
    true_optimum = int(deterministic_order(reference, selection, candidate_ids)[0])
    rows = []
    if nomination is None:
        for budget in BUDGETS:
            row = template.to_dict()
            row.update({
                "nomination_type": nomination_type,
                "status": status,
                "budget": budget,
                "nomination_index": delivery.get("candidate_index"),
                "nomination_candidate_id": delivery.get("candidate_id"),
                "true_optimum_index": true_optimum,
                "g_verifier": np.nan,
                "g_total": np.nan,
                "verifier_failure": pd.NA,
                "g_search_proxy": np.nan,
                "witness_rank": np.nan,
                "l_b": np.nan,
                "l_b_over_g_verifier": np.nan,
                "certified": pd.NA,
            })
            rows.append(row)
        return rows
    exact = int(proxy_order[0])
    probe_order = np.concatenate(([nomination], proxy_order[proxy_order != nomination]))
    regret = max(0.0, float(reference[true_optimum] - reference[nomination]))
    witnesses = np.flatnonzero(reference[probe_order] > reference[nomination] + TOLERANCE)
    witness_rank = int(witnesses[0] + 1) if len(witnesses) else None
    for budget in BUDGETS:
        trusted = probe_order[: min(budget, len(probe_order))]
        lower = max(0.0, float(reference[trusted].max() - reference[nomination]))
        row = template.to_dict()
        row.update({
            "nomination_type": nomination_type,
            "status": status,
            "budget": budget,
            "nomination_index": nomination,
            "nomination_candidate_id": int(candidate_ids[nomination]),
            "true_optimum_index": true_optimum,
            "exact_learned_score_optimum_index": exact,
            "g_verifier": regret,
            "g_total": regret,
            "verifier_failure": regret > TOLERANCE,
            "g_search_proxy": max(0.0, float(prediction[exact] - prediction[nomination])),
            "witness_rank": witness_rank,
            "l_b": lower,
            "l_b_over_g_verifier": lower / regret if regret > TOLERANCE else np.nan,
            "certified": lower > TOLERANCE,
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("ram", "com"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    defaults = {"ram": OPTIMIZERS_RAM, "com": OPTIMIZERS_COM}
    output = args.output_dir if args.output_dir is not None else defaults[args.method]
    config = json.loads((output / "config.json").read_text())
    old = pd.read_parquet(output / "metrics.parquet")
    exact = old[old.nomination_type == "exact_learned_score_optimum"].copy()
    if "best_eligible_method_proposal" in set(old.nomination_type):
        eligible = old[old.nomination_type == "best_eligible_method_proposal"].copy()
    else:
        eligible = old[old.nomination_type == "method_delivered"].copy()
        eligible["nomination_type"] = "best_eligible_method_proposal"
    delivered_rows: list[dict[str, object]] = []
    delivery_audit = []
    for dataset in config["datasets"]:
        _, tokens, reference, candidate_ids = load_dataset(dataset)
        for seed in config["seeds"]:
            split = load_split(output / "raw" / "splits" / f"{dataset}_seed_{seed}.npz")
            proposal_path = output / "raw" / "proposals" / f"{dataset}_seed_{seed}.parquet"
            proposals = pd.read_parquet(proposal_path)
            scores = (
                ram_scores(output, dataset, seed, tokens, split, config)
                if args.method == "ram"
                else com_scores(output, dataset, seed, tokens, split, config)
            )
            if len(scores) != len(proposals) or not np.all(np.isfinite(scores)):
                raise AssertionError("invalid proposal learned scores")
            proposals["learned_score"] = scores
            order = np.lexsort((proposals.proposal_number.to_numpy(), -scores))
            winner_position = int(order[0])
            proposals["is_true_method_delivery"] = False
            proposals.loc[winner_position, "is_true_method_delivery"] = True
            proposals["is_selection_eligible"] = proposals.split == "selection"
            proposals.to_parquet(proposal_path, index=False)
            winner = proposals.iloc[winner_position]
            eligible_subset = proposals[proposals.split == "selection"]
            if eligible_subset.empty:
                best_eligible_number = None
            else:
                eligible_order = np.lexsort((eligible_subset.proposal_number, -eligible_subset.learned_score))
                best_eligible_number = int(eligible_subset.iloc[int(eligible_order[0])].proposal_number)
            base = exact[(exact.dataset == dataset) & (exact.seed == seed)].iloc[0]
            prediction = np.load(
                output / "raw" / "predictions" / f"{dataset}_seed_{seed}.npz", allow_pickle=False
            )["prediction"].astype(np.float64)
            eligible_delivery = winner.split == "selection"
            nomination = int(winner.candidate_index) if eligible_delivery else None
            delivered_rows.extend(audit_nomination(
                base,
                "method_delivered",
                nomination,
                "complete" if eligible_delivery else "ineligible_method_delivery",
                prediction,
                reference,
                candidate_ids,
                split["selection"],
                winner,
            ))
            delivery_audit.append({
                "method": args.method,
                "dataset": dataset,
                "seed": seed,
                "proposal_count": len(proposals),
                "selection_eligible_count": int((proposals.split == "selection").sum()),
                "selection_eligible_fraction": float((proposals.split == "selection").mean()),
                "unique_selection_eligible_count": int(proposals.loc[proposals.split == "selection", "candidate_index"].nunique()),
                "true_delivery_proposal_number": int(winner.proposal_number),
                "true_delivery_learned_score": float(winner.learned_score),
                "true_delivery_tokens": winner.tokens,
                "true_delivery_candidate_index": winner.candidate_index,
                "true_delivery_candidate_id": winner.candidate_id,
                "true_delivery_split": winner.split,
                "true_delivery_selection_eligible": bool(eligible_delivery),
                "best_eligible_proposal_number": best_eligible_number,
            })
    frame = pd.concat([exact, eligible, pd.DataFrame(delivered_rows)], ignore_index=True)
    frame = frame.sort_values(["dataset", "seed", "nomination_type", "budget"]).reset_index(drop=True)
    frame.to_parquet(output / "metrics.parquet", index=False)
    frame.to_parquet(output / "raw_results.parquet", index=False)
    delivery = pd.DataFrame(delivery_audit)
    delivery.to_parquet(output / "method_delivery_audit.parquet", index=False)
    diagnostics_path = output / "diagnostics.parquet"
    diagnostics = pd.read_parquet(diagnostics_path)
    diagnostics = diagnostics.rename(columns={
        "method_delivered_nomination_index": "best_eligible_method_proposal_index"
    })
    diagnostics = diagnostics.drop(columns=[
        "true_method_delivery_candidate_index",
        "true_method_delivery_split",
        "true_method_delivery_selection_eligible",
        "selection_eligible_proposal_count",
        "selection_eligible_proposal_fraction",
    ], errors="ignore")
    delivery_columns = delivery[[
        "dataset", "seed", "true_delivery_candidate_index", "true_delivery_split",
        "true_delivery_selection_eligible", "selection_eligible_count",
        "selection_eligible_fraction",
    ]].rename(columns={
        "true_delivery_candidate_index": "true_method_delivery_candidate_index",
        "true_delivery_split": "true_method_delivery_split",
        "true_delivery_selection_eligible": "true_method_delivery_selection_eligible",
        "selection_eligible_count": "selection_eligible_proposal_count",
        "selection_eligible_fraction": "selection_eligible_proposal_fraction",
    })
    diagnostics = diagnostics.merge(delivery_columns, on=["dataset", "seed"], how="left", validate="one_to_one")
    diagnostics.to_parquet(diagnostics_path, index=False)
    config["nomination_semantics"] = {
        "method_delivered": "highest learned-score decoded proposal across all proposals; ineligible delivery has unavailable finite-selection regret/certificate",
        "best_eligible_method_proposal": "highest learned-score decoded proposal restricted to held-out selection; never substituted for method_delivered",
        "exact_learned_score_optimum": "exact maximum learned score over the declared finite held-out selection pool",
        "proposal_score": "pinned learned model reevaluated on exact decoded integer tokens",
    }
    config["true_delivery_selection_eligible_runs"] = int(delivery.true_delivery_selection_eligible.sum())
    config["true_delivery_ineligible_runs"] = int((~delivery.true_delivery_selection_eligible).sum())
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    with (output / "run.log").open("a") as handle:
        handle.write(json.dumps({
            "event": "method_delivery_semantic_correction",
            "runs": len(delivery),
            "eligible_true_deliveries": int(delivery.true_delivery_selection_eligible.sum()),
            "ineligible_true_deliveries": int((~delivery.true_delivery_selection_eligible).sum()),
        }, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

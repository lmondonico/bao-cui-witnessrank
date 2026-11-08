"""Run the official RaM-ListNet model on a finite landscape and audit its nominations."""

from __future__ import annotations

import argparse
import copy
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.published_optimizers.common import (
    BUDGETS,
    DATASETS,
    SEEDS,
    audit_predictions,
    design_bench_logits,
    environment_record,
    logits_to_integers,
    load_dataset,
    make_split,
    normalize_from_train,
    proposal_indices,
    proposal_diagnostics,
    save_split,
    stable_seed,
    write_json,
    write_results_markdown,
)
from experiments.paths import OPTIMIZERS_RAM, ROOT


RAM_ROOT = ROOT / ".external" / "ram"
OUTPUT = OPTIMIZERS_RAM
REPO_URL = "https://github.com/lamda-bbo/Offline-RaM.git"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def native_sanity() -> dict[str, object]:
    command = [
        sys.executable,
        "main_from_scratch.py",
        "--task",
        "TFBind8-Exact-v0",
        "--loss",
        "listnet",
        "--seed",
        "1",
        "--num-samples",
        "128",
        "--list-length",
        "32",
        "--batch-size",
        "32",
    ]
    completed = subprocess.run(command, cwd=RAM_ROOT, text=True, capture_output=True, check=False)
    return {
        "name": "official_native_entrypoint_reduced_tfbind8",
        "command": command,
        "cwd": str(RAM_ROOT),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "expected_documented_environment": "Python 3.8, torch 1.13.1, design-bench 2.0.12",
    }


def theory_reference_sanity() -> dict[str, object]:
    destination = OUTPUT / "raw" / "reference_sanity"
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(RAM_ROOT)
    environment["MPLBACKEND"] = "Agg"
    command = [
        sys.executable,
        str(RAM_ROOT / "theory_simulation" / "simulation_example.py"),
        "--seed",
        "42",
    ]
    completed = subprocess.run(
        command, cwd=destination, text=True, capture_output=True, check=False, env=environment
    )
    return {
        "name": "official_theory_simulation_example",
        "command": command,
        "cwd": str(destination),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": (
            completed.returncode == 0
            and (destination / "Simulation.pdf").exists()
            and "nan" not in completed.stdout.lower()
        ),
        "scientific_failure": "nan" in completed.stdout.lower(),
    }


def component_sanity(model_module, loss_module) -> dict[str, object]:
    torch.manual_seed(7)
    model = model_module.SimpleMLP(input_dim=3, hidden_dim=[8, 8], output_dim=1)
    loss = loss_module.ListNetLossFunction()
    x = torch.tensor(
        [[[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 1.0, 0.0]]], dtype=torch.float32
    )
    y = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    before = float(loss(model(x).squeeze(-1), y).detach())
    for _ in range(30):
        value = loss(model(x).squeeze(-1), y)
        optimizer.zero_grad()
        value.backward()
        optimizer.step()
    after = float(loss(model(x).squeeze(-1), y).detach())
    return {
        "name": "unchanged_official_simple_mlp_listnet_update",
        "passed": np.isfinite(after) and after < before,
        "loss_before": before,
        "loss_after": after,
        "source_model": str(RAM_ROOT / "model.py"),
        "source_loss": str(RAM_ROOT / "losses" / "listnet.py"),
    }


def predict(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    result = np.empty(len(features), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 32768):
            stop = min(start + 32768, len(features))
            batch = torch.from_numpy(features[start:stop]).to(device)
            result[start:stop] = model(batch).squeeze(-1).cpu().numpy()
    return result


def train_one(
    *,
    name: str,
    seed: int,
    model_module,
    loss_module,
    search_module,
    epochs: int,
    list_length: int,
    lists_per_epoch: int,
    hidden_size: int,
    search_solutions: int,
    search_steps: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.monotonic()
    _, tokens, reference, candidate_ids = load_dataset(name)
    split = make_split(name, candidate_ids, seed)
    save_split(OUTPUT / "raw" / "splits" / f"{name}_seed_{seed}.npz", split, candidate_ids)
    alphabet_size = DATASETS[name]["alphabet_size"]
    logits = design_bench_logits(tokens, alphabet_size)
    flattened = logits.reshape(len(logits), -1)
    features, feature_mean, feature_scale = normalize_from_train(flattened, split["train"])
    target_mean = float(reference[split["train"]].mean())
    target_scale = float(reference[split["train"]].std())
    if target_scale < 1e-12:
        raise AssertionError("degenerate training labels")
    target = ((reference - target_mean) / target_scale).astype(np.float32)

    run_seed = stable_seed("ram-official", name, seed)
    np_rng = np.random.default_rng(run_seed)
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_module.SimpleMLP(
        input_dim=features.shape[1], hidden_dim=[hidden_size, hidden_size], output_dim=1
    ).to(device)
    loss_fn = loss_module.ListNetLossFunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_validation = float("inf")
    best_state = None
    history: list[dict[str, object]] = []
    train = split["train"]
    validation = split["validation"]
    validation_lists = np_rng.choice(validation, size=(min(64, lists_per_epoch), list_length), replace=True)
    validation_x = torch.from_numpy(features[validation_lists]).to(device)
    validation_y = torch.from_numpy(target[validation_lists]).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for _ in range(lists_per_epoch):
            indices = np_rng.choice(train, size=list_length, replace=False)
            x_batch = torch.from_numpy(features[indices][None]).to(device)
            y_batch = torch.from_numpy(target[indices][None]).to(device)
            value = loss_fn(model(x_batch).squeeze(-1), y_batch)
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
            epoch_losses.append(float(value.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_fn(model(validation_x).squeeze(-1), validation_y).detach().cpu()
            )
        history.append({
            "dataset": name,
            "seed": seed,
            "epoch": epoch,
            "train_listnet": float(np.mean(epoch_losses)),
            "validation_listnet": validation_loss,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise AssertionError("no RAM checkpoint")
    model.load_state_dict(best_state)
    raw_prediction = predict(model, features, device)
    prediction = target_mean + target_scale * raw_prediction
    validation_rmse = float(
        np.sqrt(np.mean((prediction[validation] - reference[validation]) ** 2))
    )

    training_best = train[np.lexsort((candidate_ids[train], -reference[train]))[:search_solutions]]
    x_init = torch.from_numpy(features[training_best]).to(device)
    searched = search_module.adam_search(
        x_init=x_init,
        forward_model=model,
        score_fn=lambda value: value,
        x_opt_lr=0.1,
        x_opt_step=search_steps,
    )
    searched_flat = searched.detach().cpu().numpy() * feature_scale + feature_mean
    searched_logits = searched_flat.reshape((-1, tokens.shape[1], alphabet_size - 1))
    searched_tokens = logits_to_integers(searched_logits)
    delivered, ineligible = proposal_indices(searched_tokens, tokens, split["selection"])
    proposal_frame = pd.DataFrame(proposal_diagnostics(
        searched_tokens, tokens, split, candidate_ids
    ))

    output_path = OUTPUT / "raw" / "predictions" / f"{name}_seed_{seed}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prediction=prediction.astype(np.float32),
        proposal_tokens=searched_tokens.astype(np.int8),
        eligible_proposal_indices=delivered,
        best_validation_listnet=best_validation,
        validation_rmse=validation_rmse,
    )
    proposal_path = OUTPUT / "raw" / "proposals" / f"{name}_seed_{seed}.parquet"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_frame.to_parquet(proposal_path, index=False)
    checkpoint_path = OUTPUT / "raw" / "checkpoints" / f"{name}_seed_{seed}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint_path)
    history_frame = pd.DataFrame(history)
    history_path = OUTPUT / "raw" / "training" / f"{name}_seed_{seed}.parquet"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_frame.to_parquet(history_path, index=False)

    return audit_predictions(
        method="official_offline_ram_listnet",
        dataset=name,
        seed=seed,
        prediction=prediction,
        reference=reference,
        candidate_ids=candidate_ids,
        selection=split["selection"],
        delivered=delivered,
        proposal_count=len(searched_tokens),
        ineligible_proposal_count=ineligible,
        validation_rmse=validation_rmse,
        training_seconds=time.monotonic() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--list-length", type=int, default=256)
    parser.add_argument("--lists-per-epoch", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--search-solutions", type=int, default=128)
    parser.add_argument("--search-steps", type=int, default=100)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT / "run.log"
    sys.path.insert(0, str(RAM_ROOT))
    model_module = importlib.import_module("model")
    loss_module = importlib.import_module("losses.listnet")
    search_module = importlib.import_module("search")
    sanity = [
        native_sanity(),
        theory_reference_sanity(),
        component_sanity(model_module, loss_module),
    ]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with log_path.open("w") as log:
        log.write(json.dumps({"event": "start", **environment_record()}, sort_keys=True) + "\n")
        for check in sanity:
            log.write(json.dumps({"event": "sanity", **check}, sort_keys=True) + "\n")
        for dataset in args.datasets:
            for seed in args.seeds:
                try:
                    result, diagnostic = train_one(
                        name=dataset,
                        seed=seed,
                        model_module=model_module,
                        loss_module=loss_module,
                        search_module=search_module,
                        epochs=args.epochs,
                        list_length=args.list_length,
                        lists_per_epoch=args.lists_per_epoch,
                        hidden_size=args.hidden_size,
                        search_solutions=args.search_solutions,
                        search_steps=args.search_steps,
                    )
                    rows.extend(result)
                    diagnostics.append(diagnostic)
                    log.write(json.dumps({"event": "complete", **diagnostic}, sort_keys=True) + "\n")
                    log.flush()
                except Exception as error:
                    failure = {
                        "dataset": dataset,
                        "seed": seed,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    failures.append(failure)
                    log.write(json.dumps({"event": "failure", **failure}, sort_keys=True) + "\n")
                    log.flush()
    frame = pd.DataFrame(rows)
    diagnostic_frame = pd.DataFrame(diagnostics)
    frame.to_parquet(OUTPUT / "metrics.parquet", index=False)
    frame.to_parquet(OUTPUT / "raw_results.parquet", index=False)
    diagnostic_frame.to_parquet(OUTPUT / "diagnostics.parquet", index=False)
    config = {
        **vars(args),
        "method": "Official Offline-RaM ListNet",
        "hyperparameter_source": "official config/default.yaml with declared compute reductions; same adapter settings across datasets",
        "budgets": list(BUDGETS),
        "repository_url": REPO_URL,
        "input_transform": "Exact Design-Bench discrete logits (soft_interpolation=0.6), then training-split standardization",
        "target_transform": "training-split standardization",
        "official_components_unmodified": ["model.py:SimpleMLP", "losses/listnet.py:ListNetLossFunction", "search.py:adam_search"],
        "adapter_modifications": [
            "local finite Parquet loader replaces Design-Bench Task",
            "explicit persisted train/validation/selection split",
            "author architecture reduced from [2048,2048] to configured hidden size",
            "author 10000 lists x 1000 entries x 100 epochs reduced to configured online list sampling",
            "method proposals are mapped back to exact finite candidate tokens and filtered to selection pool",
        ],
        "failed_runs": failures,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "environment": environment_record(),
    }
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "provenance.json", {
        "repository_url": REPO_URL,
        "paper": "Offline Model-Based Optimization by Learning to Rank (ICLR 2025)",
        "official_status": "Repository README states official implementation",
        "environment": environment_record(),
    })
    write_json(OUTPUT / "sanity_checks.json", sanity)
    write_results_markdown(
        OUTPUT / "results.md",
        "Offline-RaM ListNet",
        frame,
        "The exact official model, ListNet loss, and Adam design optimizer were imported from the pinned author repository. "
        "The finite-data adapter and all compute reductions are declared in `config.json`. This is an official-component "
        "integration, not a faithful published-scale reproduction.",
    )


if __name__ == "__main__":
    main()

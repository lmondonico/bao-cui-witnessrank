"""Run the official COM trainer on a finite landscape and audit its nominations."""

from __future__ import annotations

import argparse
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
    proposal_diagnostics,
    proposal_indices,
    save_split,
    stable_seed,
    write_json,
    write_results_markdown,
)
from experiments.paths import OPTIMIZERS_COM, ROOT


COM_ROOT = ROOT / ".external" / "com"
OUTPUT = OPTIMIZERS_COM
REPO_URL = "https://github.com/brandontrabucco/design-baselines.git"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def native_sanity() -> dict[str, object]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{COM_ROOT / 'python'}:{COM_ROOT}"
    command = [
        sys.executable, "-c",
        "from design_baselines.coms_cleaned import coms_cleaned; coms_cleaned()",
        "--help",
    ]
    completed = subprocess.run(
        command, cwd=COM_ROOT, text=True, capture_output=True, check=False, env=environment
    )
    return {
        "name": "official_native_entrypoint_help",
        "command": command,
        "cwd": str(COM_ROOT),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "expected_documented_environment": "Python >=3.7,<3.8, TensorFlow 2.4.0, design-bench 2.0.12",
    }


def install_keras3_mse_alias(tf) -> bool:
    """Compatibility for the removed TF 2.4 functional MSE symbol."""
    if hasattr(tf.keras.losses, "mean_squared_error"):
        return False

    def mean_squared_error(y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)

    tf.keras.losses.mean_squared_error = mean_squared_error
    return True


def component_sanity(tf, nets_module, trainers_module) -> dict[str, object]:
    tf.keras.utils.set_random_seed(7)
    model = nets_module.ForwardModel((4,), activations=("relu", "relu"), hidden_size=16)
    trainer = trainers_module.ConservativeObjectiveModel(
        model,
        forward_model_lr=3e-4,
        alpha=0.1,
        alpha_lr=0.01,
        overestimation_limit=0.5,
        particle_lr=0.05,
        particle_gradient_steps=2,
        entropy_coefficient=0.0,
        noise_std=0.0,
    )
    x = tf.random.normal((8, 4))
    y = tf.reduce_sum(x, axis=1, keepdims=True)
    statistics = trainer.train_step(x, y)
    optimized = trainer.optimize(x, 2)
    values = {key: float(tf.reduce_mean(value)) for key, value in statistics.items()}
    return {
        "name": "unchanged_official_com_gradient_update",
        "passed": all(np.isfinite(value) for value in values.values()) and optimized.shape == x.shape,
        "statistics": values,
        "source_model": str(COM_ROOT / "design_baselines" / "coms_cleaned" / "nets.py"),
        "source_trainer": str(COM_ROOT / "design_baselines" / "coms_cleaned" / "trainers.py"),
    }


def aggregation_wrapper_sanity(tf, nets_module, trainers_module) -> dict[str, object]:
    tf.keras.utils.set_random_seed(8)
    model = nets_module.ForwardModel((4,), activations=("relu", "relu"), hidden_size=16)
    trainer = trainers_module.ConservativeObjectiveModel(
        model,
        forward_model_lr=3e-4,
        alpha=0.1,
        alpha_lr=0.01,
        overestimation_limit=0.5,
        particle_lr=0.05,
        particle_gradient_steps=1,
        entropy_coefficient=0.0,
        noise_std=0.0,
    )
    x = tf.random.normal((8, 4))
    y = tf.reduce_sum(x, axis=1, keepdims=True)
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(4)
    try:
        trainer.train(dataset)
    except Exception as error:
        return {
            "name": "official_train_aggregation_wrapper_tf220",
            "passed": False,
            "error": repr(error),
            "adapter_response": "call unchanged train_step per batch and aggregate batch means",
        }
    return {"name": "official_train_aggregation_wrapper_tf220", "passed": True}


def tensor_dataset(tf, features: np.ndarray, target: np.ndarray, indices: np.ndarray, batch_size: int, seed: int, shuffle: bool):
    dataset = tf.data.Dataset.from_tensor_slices((features[indices], target[indices, None]))
    if shuffle:
        dataset = dataset.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    dataset = dataset.with_options(options).batch(batch_size)
    return dataset.prefetch(2)


def predict(model, features: np.ndarray) -> np.ndarray:
    result = np.empty(len(features), dtype=np.float64)
    for start in range(0, len(features), 32768):
        stop = min(start + 32768, len(features))
        result[start:stop] = model(features[start:stop], training=False).numpy()[:, 0]
    return result


def train_one(
    *,
    name: str,
    seed: int,
    tf,
    nets_module,
    trainers_module,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    particle_train_steps: int,
    particle_evaluate_steps: int,
    particle_lr: float,
    overestimation_limit: float,
    evaluation_samples: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    started = time.monotonic()
    _, tokens, reference, candidate_ids = load_dataset(name)
    split = make_split(name, candidate_ids, seed)
    save_split(OUTPUT / "raw" / "splits" / f"{name}_seed_{seed}.npz", split, candidate_ids)
    alphabet_size = DATASETS[name]["alphabet_size"]
    features = design_bench_logits(tokens, alphabet_size)
    target_mean = float(reference[split["train"]].mean())
    target_scale = float(reference[split["train"]].std())
    if target_scale < 1e-12:
        raise AssertionError("degenerate training labels")
    target = ((reference - target_mean) / target_scale).astype(np.float32)
    run_seed = stable_seed("com-official", name, seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(run_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass
    input_shape = features.shape[1:]
    model = nets_module.ForwardModel(
        input_shape,
        activations=("relu", "relu"),
        hidden_size=hidden_size,
        final_tanh=False,
    )
    effective_particle_lr = particle_lr * np.sqrt(np.prod(input_shape))
    trainer = trainers_module.ConservativeObjectiveModel(
        model,
        forward_model_lr=3e-4,
        alpha=0.1,
        alpha_lr=0.01,
        overestimation_limit=overestimation_limit,
        particle_lr=effective_particle_lr,
        particle_gradient_steps=particle_train_steps,
        entropy_coefficient=0.0,
        noise_std=0.0,
    )
    train_data = tensor_dataset(
        tf, features, target, split["train"], batch_size, run_seed, shuffle=True
    )
    validation_data = tensor_dataset(
        tf, features, target, split["validation"], batch_size, run_seed, shuffle=False
    )
    history: list[dict[str, object]] = []
    for epoch in range(1, epochs + 1):
        train_values: dict[str, list[object]] = {}
        for x_batch, y_batch in train_data:
            for key, value in trainer.train_step(x_batch, y_batch).items():
                train_values.setdefault(key, []).append(value)
        validation_values: dict[str, list[object]] = {}
        for x_batch, y_batch in validation_data:
            for key, value in trainer.validate_step(x_batch, y_batch).items():
                validation_values.setdefault(key, []).append(value)
        record: dict[str, object] = {"dataset": name, "seed": seed, "epoch": epoch}
        for prefix, statistics in (("train", train_values), ("validation", validation_values)):
            for key, values in statistics.items():
                batch_means = [tf.reduce_mean(value) for value in values]
                record[f"{prefix}:{key}"] = float(tf.reduce_mean(tf.stack(batch_means)))
        history.append(record)
        numeric = np.asarray([value for key, value in record.items() if key not in {"dataset", "seed", "epoch"}])
        if not np.all(np.isfinite(numeric)):
            raise FloatingPointError(f"non-finite COM training statistic at epoch {epoch}")

    raw_prediction = predict(model, features)
    prediction = target_mean + target_scale * raw_prediction
    validation_rmse = float(
        np.sqrt(np.mean((prediction[split["validation"]] - reference[split["validation"]]) ** 2))
    )
    train_order = split["train"][np.lexsort(
        (candidate_ids[split["train"]], -reference[split["train"]])
    )]
    initial = features[train_order[:evaluation_samples]]
    optimized = trainer.optimize(tf.convert_to_tensor(initial), particle_evaluate_steps, training=False).numpy()
    proposal_tokens = logits_to_integers(optimized)
    delivered, ineligible = proposal_indices(proposal_tokens, tokens, split["selection"])
    proposal_frame = pd.DataFrame(proposal_diagnostics(
        proposal_tokens, tokens, split, candidate_ids
    ))

    output_path = OUTPUT / "raw" / "predictions" / f"{name}_seed_{seed}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        prediction=prediction.astype(np.float32),
        proposal_tokens=proposal_tokens.astype(np.int8),
        eligible_proposal_indices=delivered,
        validation_rmse=validation_rmse,
    )
    proposal_path = OUTPUT / "raw" / "proposals" / f"{name}_seed_{seed}.parquet"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_frame.to_parquet(proposal_path, index=False)
    checkpoint_path = OUTPUT / "raw" / "checkpoints" / f"{name}_seed_{seed}.weights.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(checkpoint_path)
    history_path = OUTPUT / "raw" / "training" / f"{name}_seed_{seed}.parquet"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_parquet(history_path, index=False)
    return audit_predictions(
        method="official_conservative_objective_model",
        dataset=name,
        seed=seed,
        prediction=prediction,
        reference=reference,
        candidate_ids=candidate_ids,
        selection=split["selection"],
        delivered=delivered,
        proposal_count=len(proposal_tokens),
        ineligible_proposal_count=ineligible,
        validation_rmse=validation_rmse,
        training_seconds=time.monotonic() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--particle-train-steps", type=int, default=20)
    parser.add_argument("--particle-evaluate-steps", type=int, default=50)
    parser.add_argument("--particle-lr", type=float, default=2.0)
    parser.add_argument("--overestimation-limit", type=float, default=2.0)
    parser.add_argument("--evaluation-samples", type=int, default=128)
    args = parser.parse_args()
    sys.path.insert(0, str(COM_ROOT))
    import tensorflow as tf
    import tensorflow_probability as tfp

    compatibility_alias_installed = install_keras3_mse_alias(tf)
    nets_module = load_module(
        "official_com_nets", COM_ROOT / "design_baselines" / "coms_cleaned" / "nets.py"
    )
    trainers_module = load_module(
        "official_com_trainers", COM_ROOT / "design_baselines" / "coms_cleaned" / "trainers.py"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sanity = [
        native_sanity(),
        aggregation_wrapper_sanity(tf, nets_module, trainers_module),
        component_sanity(tf, nets_module, trainers_module),
    ]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with (OUTPUT / "run.log").open("w") as log:
        log.write(json.dumps({"event": "start", **environment_record()}, sort_keys=True) + "\n")
        for check in sanity:
            log.write(json.dumps({"event": "sanity", **check}, sort_keys=True) + "\n")
        for dataset in args.datasets:
            for seed in args.seeds:
                try:
                    result, diagnostic = train_one(
                        name=dataset,
                        seed=seed,
                        tf=tf,
                        nets_module=nets_module,
                        trainers_module=trainers_module,
                        epochs=args.epochs,
                        hidden_size=args.hidden_size,
                        batch_size=args.batch_size,
                        particle_train_steps=args.particle_train_steps,
                        particle_evaluate_steps=args.particle_evaluate_steps,
                        particle_lr=args.particle_lr,
                        overestimation_limit=args.overestimation_limit,
                        evaluation_samples=args.evaluation_samples,
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
        "method": "Official Conservative Objective Models",
        "hyperparameter_source": "official scripts/coms-tf-bind-8.sh; reused without dataset-specific tuning for TrpB and GB1",
        "budgets": list(BUDGETS),
        "repository_url": REPO_URL,
        "input_transform": "Exact Design-Bench discrete logits at author default soft_interpolation=0.6; no x normalization",
        "target_transform": "training-split standardization",
        "official_components_unmodified": [
            "design_baselines/coms_cleaned/nets.py:ForwardModel",
            "design_baselines/coms_cleaned/trainers.py:ConservativeObjectiveModel",
        ],
        "adapter_modifications": [
            "local finite Parquet loader replaces StaticGraphTask",
            "explicit persisted train/validation/selection split",
            "Keras-3 alias restores removed tf.keras.losses.mean_squared_error with its historical formula",
            "TF2.20-incompatible train/validate scalar-concat logging wrapper is bypassed; unchanged train_step/validate_step are called per batch",
            "author hidden size 2048, 50 epochs, and 50 inner steps reduced to configured values",
            "method proposals are mapped back with exact Design-Bench argmax inverse and filtered to selection pool",
        ],
        "keras3_mse_alias_installed": compatibility_alias_installed,
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "failed_runs": failures,
        "environment": environment_record(),
    }
    write_json(OUTPUT / "config.json", config)
    write_json(OUTPUT / "provenance.json", {
        "repository_url": REPO_URL,
        "paper": "Conservative Objective Models for Effective Offline Model-Based Optimization (ICML 2021)",
        "official_status": "COM project page links this author repository as Code",
        "environment": environment_record(),
        "isolated_dependencies": str(COM_ROOT / "python"),
    })
    write_json(OUTPUT / "sanity_checks.json", sanity)
    write_results_markdown(
        OUTPUT / "results.md",
        "Conservative Objective Models",
        frame,
        "The exact official forward model and conservative trainer were imported from the pinned author repository. "
        "The finite-data adapter, Keras compatibility alias, and compute reductions are declared in `config.json`. "
        "This is an official-component integration, not a faithful published-scale reproduction.",
    )


if __name__ == "__main__":
    main()

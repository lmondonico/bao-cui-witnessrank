"""COM runs at the published hyperparameters."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from experiments.paths import OPTIMIZERS_COM
from experiments.published_optimizers.com import adapter as runner


OUTPUT = OPTIMIZERS_COM


def native_sanity() -> dict[str, object]:
    """Attempt the author CLI with published TF-Bind-8 method settings."""
    destination = OUTPUT / "raw" / "native_attempt"
    destination.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join([
        str((runner.COM_ROOT / "python_legacy").resolve()),
        str((runner.COM_ROOT / "python").resolve()),
        str(runner.COM_ROOT.resolve()),
    ])
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    command = [
        sys.executable, "-c",
        "from design_baselines.coms_cleaned import coms_cleaned; coms_cleaned()",
        "--logging-dir", str(destination.resolve()),
        "--task", "TFBind8-Exact-v0", "--no-task-relabel", "--normalize-ys",
        "--no-normalize-xs", "--not-in-latent-space", "--particle-lr", "2.0",
        "--particle-train-gradient-steps", "50", "--particle-evaluate-gradient-steps", "50",
        "--particle-entropy-coefficient", "0.0", "--forward-model-activations", "relu",
        "--forward-model-activations", "relu", "--forward-model-hidden-size", "2048",
        "--no-forward-model-final-tanh", "--forward-model-lr", "0.0003",
        "--forward-model-alpha", "0.1", "--forward-model-alpha-lr", "0.01",
        "--forward-model-overestimation-limit", "2.0", "--forward-model-noise-std", "0.0",
        "--forward-model-batch-size", "128", "--forward-model-val-size", "500",
        "--forward-model-epochs", "50", "--evaluation-samples", "128", "--fast",
    ]
    completed = subprocess.run(
        command, cwd=runner.COM_ROOT, env=environment, text=True, capture_output=True, check=False
    )
    return {
        "name": "official_native_tfbind8_published_cli_isolated",
        "command": command,
        "cwd": str(runner.COM_ROOT),
        "pythonpath": environment["PYTHONPATH"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "expected_documented_environment": "Python >=3.7,<3.8, TensorFlow 2.4.0, design-bench 2.0.12",
        "actual_compatibility_environment": "Python 3.12, TensorFlow 2.20, isolated design-bench 2.0.12",
    }


def main() -> None:
    runner.OUTPUT = OUTPUT
    runner.native_sanity = native_sanity
    runner.main()
    path = OUTPUT / "config.json"
    config = json.loads(path.read_text())
    config.update({
        "artifact_track": "official_components",
        "scale_classification": "official-component finite-landscape adaptation at the published TF-Bind-8 scale",
        "published_tf_bind_8_script": {
            "hidden_size": 2048,
            "epochs": 50,
            "batch_size": 128,
            "particle_train_gradient_steps": 50,
            "particle_evaluate_gradient_steps": 50,
            "particle_lr": 2.0,
            "overestimation_limit": 2.0,
            "evaluation_samples": 128,
        },
        "native_default_match": {
            "hidden_size": config["hidden_size"] == 2048,
            "epochs": config["epochs"] == 50,
            "batch_size": config["batch_size"] == 128,
            "particle_train_steps": config["particle_train_steps"] == 50,
            "particle_evaluate_steps": config["particle_evaluate_steps"] == 50,
            "particle_lr": config["particle_lr"] == 2.0,
            "overestimation_limit": config["overestimation_limit"] == 2.0,
            "evaluation_samples": config["evaluation_samples"] == 128,
        },
        "isolated_compatibility_environment": {
            "modern_tensorflow": str(runner.COM_ROOT / "python"),
            "design_bench_2_0_12_attempt": str(runner.COM_ROOT / "python_legacy"),
        },
    })
    config["adapter_modifications"] = [
        "local finite Parquet loader replaces StaticGraphTask",
        "explicit persisted train/validation/selection split",
        "Keras-3 alias restores removed tf.keras.losses.mean_squared_error using its historical formula",
        "TF2.20-incompatible scalar-concat aggregation wrapper is bypassed; unchanged train_step/validate_step run per batch",
        "method proposals use the exact Design-Bench argmax inverse; invalid candidates are retained and never projected",
        "TF-Bind-8 published hyperparameters are reused without dataset-specific tuning for TrpB and GB1",
    ]
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

"""RaM-ListNet runs at the published width, epochs, list length, and search scale."""

from __future__ import annotations

import json
import os
import subprocess
import sys

from experiments.paths import OPTIMIZERS_RAM
from experiments.published_optimizers.ram import adapter as runner


OUTPUT = OPTIMIZERS_RAM


def native_sanity() -> dict[str, object]:
    """Attempt the author entry point in the isolated Design-Bench environment."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join([
        str((runner.RAM_ROOT / "python_legacy").resolve()),
        str((runner.ROOT / ".external" / "com" / "python").resolve()),
        str(runner.RAM_ROOT.resolve()),
    ])
    environment["WANDB_MODE"] = "disabled"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"
    command = [
        sys.executable, "main_from_scratch.py", "--task", "TFBind8-Exact-v0",
        "--loss", "listnet", "--seed", "1", "--num-samples", "128",
        "--list-length", "32", "--batch-size", "32",
    ]
    completed = subprocess.run(
        command, cwd=runner.RAM_ROOT, env=environment, text=True, capture_output=True, check=False
    )
    return {
        "name": "official_native_entrypoint_isolated_reduced_tfbind8",
        "command": command,
        "cwd": str(runner.RAM_ROOT),
        "pythonpath": environment["PYTHONPATH"],
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "passed": completed.returncode == 0,
        "expected_documented_environment": "Python 3.8, torch 1.13.1, design-bench 2.0.12",
        "actual_compatibility_environment": "Python 3.12, system torch, isolated design-bench 2.0.12, modern TensorFlow",
    }


def main() -> None:
    runner.OUTPUT = OUTPUT
    runner.native_sanity = native_sanity
    runner.main()
    path = OUTPUT / "config.json"
    config = json.loads(path.read_text())
    config.update({
        "artifact_track": "official_components",
        "scale_classification": "official-component adaptation at the largest practical declared scale",
        "published_default": {
            "hidden_dim": [2048, 2048],
            "epochs": 100,
            "list_length": 1000,
            "fixed_lists": 10000,
            "list_batch_size": 128,
            "search_solutions": 128,
            "search_steps_discrete": 100,
        },
        "native_default_match": {
            "hidden_dim": config["hidden_size"] == 2048,
            "epochs": config["epochs"] == 100,
            "list_length": config["list_length"] == 1000,
            "search_solutions": config["search_solutions"] == 128,
            "search_steps": config["search_steps"] == 100,
            "fixed_list_corpus_and_batching": False,
        },
        "list_sampling_adaptation": (
            "Lists are sampled online one list per optimizer update. The official code constructs "
            "10,000 fixed lists and reuses them for 100 epochs in batches of 128."
        ),
        "list_scale_accounting": {
            "adapted_train_list_presentations": config["epochs"] * config["lists_per_epoch"],
            "published_train_list_presentations": 100 * 8000,
            "published_validation_list_presentations": 100 * 2000,
            "adapted_to_published_train_fraction": (
                config["epochs"] * config["lists_per_epoch"] / (100 * 8000)
            ),
            "note": "Published counts reflect the official 80/20 split of 10,000 fixed lists; adapter validation uses a separate declared candidate pool.",
        },
        "isolated_compatibility_environment": str(runner.RAM_ROOT / "python_legacy"),
    })
    config["adapter_modifications"] = [
        "local finite Parquet loader replaces Design-Bench Task",
        "explicit persisted train/validation/selection split",
        "online one-list optimizer updates replace the official fixed 10,000-list corpus and 128-list batches",
        "method proposals use the exact Design-Bench argmax inverse; invalid candidates are retained and never projected",
    ]
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

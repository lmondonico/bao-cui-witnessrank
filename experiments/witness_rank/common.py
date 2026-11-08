"""Load the 230 saved audits (predictions, splits, and trusted scores) for the analyses."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.paths import RESULTS as OUTPUT
from experiments.paths import DATA, AUDITS_INITIAL, AUDITS_PRIMARY
from experiments.paths import OPTIMIZERS_COM, OPTIMIZERS_RAM, ROOT

BUDGETS = (8, 16, 32, 64, 128, 256)
MULTIPLIERS = (0.0, 0.01, 0.025, 0.05, 0.10, 0.25)
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Instance:
    instance_id: str
    source: str
    landscape: str
    model: str
    split: int
    prediction_path: Path
    split_path: Path
    persisted_g: float
    persisted_r: int


@dataclass(frozen=True)
class LoadedInstance:
    metadata: Instance
    candidate_ids: np.ndarray
    reference: np.ndarray
    prediction: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    selection: np.ndarray
    order: np.ndarray


def stable_seed(*values: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, values)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def ids_to_indices(candidate_ids: np.ndarray, requested: np.ndarray) -> np.ndarray:
    order = np.argsort(candidate_ids, kind="stable")
    positions = np.searchsorted(candidate_ids[order], requested)
    if np.any(positions >= len(candidate_ids)):
        raise AssertionError("split contains an unknown candidate ID")
    result = order[positions]
    if not np.array_equal(candidate_ids[result], requested):
        raise AssertionError("split contains an unknown candidate ID")
    return result.astype(np.int64)


def _core_instances() -> list[Instance]:
    external = AUDITS_PRIMARY
    index = pd.read_parquet(external / "prediction_index.parquet")
    metrics = pd.read_parquet(external / "metrics.parquet").set_index(
        ["dataset", "seed", "model"]
    )
    result: list[Instance] = []
    for row in index.itertuples(index=False):
        persisted = metrics.loc[(row.dataset, row.seed, row.model)]
        result.append(
            Instance(
                instance_id=f"core:{row.dataset}:{row.seed}:{row.model}",
                source="core",
                landscape=str(row.dataset),
                model=str(row.model),
                split=int(row.seed),
                prediction_path=ROOT / str(row.path),
                split_path=external / "raw/splits" / str(row.dataset) / f"seed_{row.seed}.npz",
                persisted_g=float(persisted.g_verifier),
                persisted_r=int(persisted.witness_rank_R),
            )
        )

    replacement = AUDITS_INITIAL / "av_gfp"
    replacement_metrics = pd.read_parquet(replacement / "metrics.parquet").set_index(
        ["dataset", "seed", "model"]
    )
    for path in sorted((replacement / "raw/predictions").glob("*.npz")):
        seed_text, model = path.stem.removeprefix("seed_").split("_", 1)
        seed = int(seed_text)
        persisted = replacement_metrics.loc[("av_gfp", seed, model)]
        result.append(
            Instance(
                instance_id=f"core:av_gfp:{seed}:{model}",
                source="core",
                landscape="av_gfp",
                model=model,
                split=seed,
                prediction_path=path,
                split_path=replacement / "splits.parquet",
                persisted_g=float(persisted.g_verifier),
                persisted_r=int(persisted.witness_rank_R),
            )
        )
    if len(result) != 200:
        raise AssertionError(f"expected 200 core audits, found {len(result)}")
    return result


def _optimizer_instances() -> list[Instance]:
    result: list[Instance] = []
    methods = {
        "ram_fullscale": OPTIMIZERS_RAM,
        "com_fullscale": OPTIMIZERS_COM,
    }
    for method, directory in methods.items():
        metrics = pd.read_parquet(directory / "metrics.parquet")
        exact = metrics[
            metrics.nomination_type.eq("exact_learned_score_optimum")
            & metrics.status.eq("complete")
        ].drop_duplicates(["dataset", "seed"])
        if len(exact) != 15:
            raise AssertionError(f"{method}: expected 15 exact-optimum runs, found {len(exact)}")
        persisted = exact.set_index(["dataset", "seed"])
        for path in sorted((directory / "raw/predictions").glob("*.npz")):
            landscape, seed_text = path.stem.rsplit("_seed_", 1)
            seed = int(seed_text)
            row = persisted.loc[(landscape, seed)]
            result.append(
                Instance(
                    instance_id=f"optimizer:{landscape}:{seed}:{method}",
                    source="optimizer",
                    landscape=landscape,
                    model=method,
                    split=seed,
                    prediction_path=path,
                    split_path=directory / "raw/splits" / f"{landscape}_seed_{seed}.npz",
                    persisted_g=float(row.g_verifier),
                    persisted_r=int(row.witness_rank),
                )
            )
    if len(result) != 30:
        raise AssertionError(f"expected 30 full-scale optimizer audits, found {len(result)}")
    return result


def instances(include_optimizers: bool = True) -> list[Instance]:
    result = _core_instances()
    if include_optimizers:
        result.extend(_optimizer_instances())
    keys = [item.instance_id for item in result]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate followups-response instance IDs")
    return sorted(result, key=lambda item: item.instance_id)


def load_instance(instance: Instance) -> LoadedInstance:
    frame = pd.read_parquet(DATA / instance.landscape / "candidates.parquet")
    candidate_ids = frame.candidate_id.to_numpy(np.int64)
    reference = frame.f.to_numpy(np.float64)
    with np.load(instance.prediction_path, allow_pickle=False) as saved:
        prediction = saved["prediction"].astype(np.float64)
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise AssertionError(f"invalid prediction array: {instance.prediction_path}")
    if instance.landscape == "av_gfp":
        split_frame = pd.read_parquet(instance.split_path)
        split_frame = split_frame[split_frame.seed.eq(instance.split)]
        split = {
            name: split_frame.loc[split_frame.split.eq(name), "candidate_index"].to_numpy(np.int64)
            for name in ("train", "validation", "selection")
        }
    else:
        with np.load(instance.split_path, allow_pickle=False) as saved:
            if "selection_indices" in saved.files:
                split = {
                    name: saved[f"{name}_indices"].astype(np.int64)
                    for name in ("train", "validation", "selection")
                }
                for name, values in split.items():
                    key = f"{name}_candidate_ids"
                    if key in saved.files and not np.array_equal(candidate_ids[values], saved[key]):
                        raise AssertionError(f"index/ID mismatch: {instance.split_path}:{name}")
            else:
                split = {
                    name: ids_to_indices(candidate_ids, saved[name].astype(np.int64))
                    for name in ("train", "validation", "selection")
                }
    all_indices = np.concatenate(tuple(split.values()))
    if len(all_indices) != len(frame) or len(np.unique(all_indices)) != len(frame):
        raise AssertionError(f"split is not an exact partition: {instance.split_path}")
    selection = split["selection"]
    order = selection[np.lexsort((candidate_ids[selection], -prediction[selection]))]
    return LoadedInstance(
        metadata=instance,
        candidate_ids=candidate_ids,
        reference=reference,
        prediction=prediction,
        train=split["train"],
        validation=split["validation"],
        selection=selection,
        order=order,
    )


def top_tail_iqr(values: np.ndarray) -> float:
    count = max(2, int(np.ceil(0.001 * len(values))))
    top = np.partition(values, len(values) - count)[-count:]
    return float(np.quantile(top, 0.75) - np.quantile(top, 0.25))


def delta_rows(loaded: LoadedInstance) -> list[dict[str, object]]:
    values = loaded.reference[loaded.selection]
    sd = float(np.std(values, ddof=0))
    iqr = top_tail_iqr(values)
    rows: list[dict[str, object]] = []
    for family, scale in (("sd", sd), ("top_tail_iqr", iqr)):
        if family == "top_tail_iqr" and loaded.metadata.landscape in {"tf_bind_8", "av_gfp"}:
            continue
        for multiplier in MULTIPLIERS:
            rows.append(
                {
                    "delta_family": family,
                    "delta_multiplier": multiplier,
                    "delta_fold": np.nan,
                    "delta": multiplier * scale,
                    "delta_scale": scale,
                    "delta_label": f"{family}:{multiplier:g}",
                    "delta_applicable": True,
                }
            )
    if loaded.metadata.landscape in {"gb1", "trpb"}:
        selected_value = float(loaded.reference[loaded.order[0]])
        for fold in (1.1, 1.25, 1.5):
            if loaded.metadata.landscape == "gb1":
                untransformed = np.expm1(selected_value)
                delta = float(np.log1p(fold * untransformed) - selected_value)
            else:
                delta = float((fold - 1.0) * selected_value) if selected_value > 0 else np.nan
            rows.append(
                {
                    "delta_family": "native_fold",
                    "delta_multiplier": np.nan,
                    "delta_fold": fold,
                    "delta": delta,
                    "delta_scale": np.nan,
                    "delta_label": f"native_fold:{fold:g}x",
                    "delta_applicable": bool(np.isfinite(delta) and delta >= 0),
                }
            )
    return rows


def is_improvement(values: np.ndarray, selected_value: float, delta: float) -> np.ndarray:
    if delta == 0.0:
        return values > selected_value + TOLERANCE
    return values >= selected_value + delta


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "No applicable rows."
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(map(str, columns)) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    rendered.append("NA")
                elif np.isposinf(value):
                    rendered.append("inf")
                elif np.isneginf(value):
                    rendered.append("-inf")
                else:
                    rendered.append(f"{value:.{digits}f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_manifest(run: str, input_paths: list[Path], seeds: dict[str, object]) -> None:
    import scipy
    import sklearn

    directory = OUTPUT / run
    directory.mkdir(parents=True, exist_ok=True)
    unique = sorted({path.resolve() for path in input_paths})
    manifest = {
        "run": run,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "packages": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "inputs": [str(path.relative_to(ROOT)) for path in unique],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def input_paths(items: list[Instance]) -> list[Path]:
    paths = [DATA / item.landscape / "candidates.parquet" for item in items]
    paths.extend(item.prediction_path for item in items)
    paths.extend(item.split_path for item in items)
    paths.extend(
        [
            AUDITS_PRIMARY / "metrics.parquet",
            AUDITS_INITIAL / "av_gfp" / "metrics.parquet",
            OPTIMIZERS_RAM / "metrics.parquet",
            OPTIMIZERS_COM / "metrics.parquet",
        ]
    )
    return paths

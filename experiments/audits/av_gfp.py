"""Prepare the avGFP landscape and run its proxy audits."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from experiments.audits.policies import (
    TOLERANCE,
    audit_gaps,
    certificate_rows,
    ranked_order,
    selection_lookup,
    sequence_search,
    top_order,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/av_gfp"
OUTPUT = ROOT / "results/initial_runs/av_gfp"
RAW = DATA / "raw/GFP_AEQVI_Sarkisyan_2016.csv"
REFERENCE = DATA / "raw/ProteinGym_reference_file_substitutions.csv"
SEEDS = tuple(range(20271201, 20271211))
BUDGETS = (8, 16, 32, 64, 128, 256)
TRAIN_SIZE = 4096
VALIDATION_SIZE = 2048
SEARCH_BUDGET = 256
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MUTATION = re.compile(r"([A-Z])(\d+)([A-Z])")


def stable_seed(*values: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, values)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**31 - 1)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def prepare() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    raw = pd.read_csv(RAW)
    references = pd.read_csv(REFERENCE)
    row = references.loc[references.DMS_id.eq("GFP_AEQVI_Sarkisyan_2016")]
    if len(row) != 1:
        raise AssertionError("missing or ambiguous ProteinGym reference record")
    wild_type = str(row.target_seq.iloc[0])
    amino_to_token = {amino: index for index, amino in enumerate(ALPHABET)}
    sequences: list[str] = []
    mutation_counts: list[int] = []
    for encoded in raw.mutant.astype(str):
        sequence = list(wild_type)
        parts = encoded.split(":")
        for part in parts:
            match = MUTATION.fullmatch(part)
            if match is None:
                raise AssertionError(f"invalid mutation: {part}")
            before, position_text, after = match.groups()
            position = int(position_text) - 1
            if sequence[position] != before:
                raise AssertionError(f"reference mismatch: {part}")
            if after not in amino_to_token:
                raise AssertionError(f"noncanonical residue: {after}")
            sequence[position] = after
        sequences.append("".join(sequence))
        mutation_counts.append(len(parts))
    if len(set(sequences)) != len(sequences):
        raise AssertionError("distinct mutation strings produce duplicate sequences")
    order = np.argsort(np.asarray(sequences), kind="stable")
    sequence_array = np.asarray(sequences, dtype=f"U{len(wild_type)}")[order]
    tokens = np.asarray(
        [[amino_to_token[amino] for amino in sequence] for sequence in sequence_array],
        dtype=np.uint8,
    )
    target = raw.DMS_score.to_numpy(np.float64)[order]
    candidate_ids = np.arange(len(raw), dtype=np.int64)
    metadata = pd.DataFrame({
        "candidate_id": candidate_ids,
        "sequence": sequence_array,
        "mutant": raw.mutant.astype(str).to_numpy()[order],
        "f": target,
        "dms_score_bin": raw.DMS_score_bin.to_numpy(np.int8)[order],
        "mutation_count": np.asarray(mutation_counts, dtype=np.int16)[order],
    })
    token_frame = pd.DataFrame(
        tokens, columns=[f"token_{position}" for position in range(tokens.shape[1])]
    )
    frame = pd.concat((metadata, token_frame), axis=1)
    audit = {
        "dataset": "GFP_AEQVI_Sarkisyan_2016",
        "trusted_score": "ProteinGym DMS_score; source phenotype mean_medianBrightness_per_aaseq; directionality +1",
        "candidate_definition": "51,714 unique measured amino-acid sequences in the pinned processed assay",
        "raw_rows": len(raw),
        "unique_mutant_strings": int(raw.mutant.nunique()),
        "unique_reconstructed_sequences": len(set(sequences)),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_mutant_strings": int(raw.mutant.duplicated().sum()),
        "conflicting_sequence_labels": 0,
        "missing_mutant": int(raw.mutant.isna().sum()),
        "missing_score": int(raw.DMS_score.isna().sum()),
        "nonfinite_score": int((~np.isfinite(raw.DMS_score)).sum()),
        "reference_mismatches": 0,
        "sequence_length": len(wild_type),
        "score_min": float(target.min()),
        "score_max": float(target.max()),
        "source_doi": "10.1038/nature17995",
        "passed": True,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DATA / "candidates.parquet", index=False)
    (DATA / "dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return frame, tokens, target, candidate_ids, audit


def mutation_features(tokens: np.ndarray, wild_tokens: np.ndarray) -> sparse.csr_matrix:
    changed_row, position = np.nonzero(tokens != wild_tokens)
    column = position * len(ALPHABET) + tokens[changed_row, position]
    values = np.ones(len(changed_row), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (changed_row, column)), shape=(len(tokens), tokens.shape[1] * len(ALPHABET))
    )


def fit_additive(
    features: sparse.csr_matrix, target: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, float, dict[str, object]]:
    choices = []
    for penalty in (1e-3, 1e-1, 10.0, 1000.0):
        model = Ridge(alpha=penalty, solver="lsqr", max_iter=5000).fit(features[train], target[train])
        validation_prediction = model.predict(features[validation])
        rmse = float(np.sqrt(np.mean((validation_prediction - target[validation]) ** 2)))
        choices.append((rmse, penalty, model))
    rmse, penalty, model = min(choices, key=lambda item: (item[0], item[1]))
    return np.asarray(model.predict(features), dtype=np.float64), rmse, {
        "model": "main-effect mutation one-hot ridge",
        "selected_alpha": penalty,
        "validation_rmse_by_alpha": {str(item[1]): item[0] for item in choices},
    }


def fit_nonlinear(
    tokens: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, dict[str, object]]:
    choices = []
    for leaf in (1, 2, 4):
        model = ExtraTreesRegressor(
            n_estimators=128,
            min_samples_leaf=leaf,
            max_features=1.0,
            random_state=seed + leaf,
            n_jobs=1,
        ).fit(tokens[train], target[train])
        validation_prediction = model.predict(tokens[validation])
        rmse = float(np.sqrt(np.mean((validation_prediction - target[validation]) ** 2)))
        choices.append((rmse, leaf, model))
    rmse, leaf, model = min(choices, key=lambda item: (item[0], item[1]))
    prediction = np.empty(len(tokens), dtype=np.float64)
    for start in range(0, len(tokens), 20_000):
        prediction[start:start + 20_000] = model.predict(tokens[start:start + 20_000])
    return prediction, rmse, {
        "model": "ExtraTrees on complete positional residue tokens",
        "n_estimators": 128,
        "n_jobs": 1,
        "selected_min_samples_leaf": leaf,
        "validation_rmse_by_leaf": {str(item[1]): item[0] for item in choices},
    }


def one_sided(
    reference: np.ndarray,
    prediction: np.ndarray,
    selection: np.ndarray,
    selected: int,
) -> dict[str, object]:
    residual_under = reference[selection] - prediction[selection]
    eps_minus = float(residual_under.max())
    e_selected = float(prediction[selected] - reference[selected])
    eps_infinity = float(np.max(np.abs(residual_under)))
    width = eps_minus + e_selected
    threshold = prediction[selected] - width
    candidate = prediction[selection] > threshold
    candidate_indices = selection[candidate]
    if selected not in set(map(int, candidate_indices)):
        candidate_indices = np.r_[candidate_indices, selected]
    witnesses = selection[reference[selection] > reference[selected] + TOLERANCE]
    contains = not len(witnesses) or bool(np.isin(witnesses, candidate_indices).all())
    if not contains or width > 2 * eps_infinity + 1e-10 or width < -1e-10:
        raise AssertionError("one-sided theorem check failed")
    return {
        "eps_minus": eps_minus,
        "e_sel": e_selected,
        "one_sided_width": width,
        "one_sided_bound_size": int(len(np.unique(candidate_indices))),
        "one_sided_contains_all_witnesses": contains,
        "one_sided_width_le_2eps": bool(width <= 2 * eps_infinity + 1e-10),
    }


def main() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "raw/predictions").mkdir(parents=True, exist_ok=True)
    if (DATA / "candidates.parquet").exists() and (DATA / "dataset_audit.json").exists():
        frame = pd.read_parquet(DATA / "candidates.parquet")
        token_columns = [f"token_{position}" for position in range(238)]
        tokens = frame[token_columns].to_numpy(np.uint8)
        reference = frame.f.to_numpy(np.float64)
        candidate_ids = frame.candidate_id.to_numpy(np.int64)
        dataset_audit = json.loads((DATA / "dataset_audit.json").read_text())
    else:
        frame, tokens, reference, candidate_ids, dataset_audit = prepare()
    reference_row = pd.read_csv(REFERENCE).loc[
        lambda value: value.DMS_id.eq("GFP_AEQVI_Sarkisyan_2016")
    ].iloc[0]
    wild_tokens = np.asarray([ALPHABET.index(amino) for amino in reference_row.target_seq], dtype=np.uint8)
    additive_features = mutation_features(tokens, wild_tokens)
    metric_rows: list[dict[str, object]] = []
    certificate_data: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    leakage_rows: list[dict[str, object]] = []
    model_details: dict[str, object] = {}
    for seed in SEEDS:
        order = np.random.default_rng(seed).permutation(len(frame))
        train = np.sort(order[:TRAIN_SIZE])
        validation = np.sort(order[TRAIN_SIZE:TRAIN_SIZE + VALIDATION_SIZE])
        selection = np.sort(order[TRAIN_SIZE + VALIDATION_SIZE:])
        sets = {name: set(map(int, candidate_ids[index])) for name, index in (
            ("train", train), ("validation", validation), ("selection", selection)
        )}
        overlaps = {
            "train_validation_overlap": len(sets["train"] & sets["validation"]),
            "train_selection_overlap": len(sets["train"] & sets["selection"]),
            "validation_selection_overlap": len(sets["validation"] & sets["selection"]),
        }
        passed = not any(overlaps.values()) and len(set.union(*sets.values())) == len(frame)
        if not passed:
            raise AssertionError(f"split leakage: {seed}")
        leakage_rows.append({"seed": seed, **overlaps, "all_candidates_once": passed, "passed": passed})
        for split_name, indices in (("train", train), ("validation", validation), ("selection", selection)):
            split_rows.extend({
                "seed": seed,
                "split": split_name,
                "candidate_index": int(index),
                "candidate_id": int(candidate_ids[index]),
            } for index in indices)
        lookup = selection_lookup(tokens, selection)
        hard = sequence_search(
            reference, tokens, selection, lookup, len(ALPHABET), SEARCH_BUDGET,
            stable_seed("av_gfp", seed, "trusted"),
        )
        fits = {}
        for model_name in ("additive_ridge", "extra_trees"):
            prediction_path = OUTPUT / "raw/predictions" / f"seed_{seed}_{model_name}.npz"
            cached = None
            if prediction_path.exists():
                with np.load(prediction_path, allow_pickle=False) as saved:
                    if saved["prediction"].dtype == np.float64:
                        cached = (
                            saved["prediction"].copy(),
                            float(saved["validation_rmse"]),
                            json.loads(str(saved["details"].item())),
                        )
            if cached is not None:
                fits[model_name] = cached
            elif model_name == "additive_ridge":
                fits[model_name] = fit_additive(additive_features, reference, train, validation)
            else:
                fits[model_name] = fit_nonlinear(
                    tokens, reference, train, validation,
                    stable_seed("av_gfp", seed, "extra_trees"),
                )
        for model_name, (prediction, validation_rmse, details) in fits.items():
            prediction_path = OUTPUT / "raw/predictions" / f"seed_{seed}_{model_name}.npz"
            np.savez_compressed(
                prediction_path,
                prediction=prediction.astype(np.float64),
                validation_rmse=validation_rmse,
                details=json.dumps(details, sort_keys=True),
            )
            model_details[f"{seed}:{model_name}"] = details
            approximate = sequence_search(
                prediction, tokens, selection, lookup, len(ALPHABET), SEARCH_BUDGET,
                stable_seed("av_gfp", seed, model_name, "proxy"),
            )
            gaps = audit_gaps(reference, prediction, selection, candidate_ids, approximate, hard)
            selected = int(gaps["x_sel_index"])
            theorem = one_sided(reference, prediction, selection, selected)
            metric_rows.append({
                "dataset": "av_gfp",
                "seed": seed,
                "model": model_name,
                "candidate_count": len(frame),
                "train_size": len(train),
                "validation_size": len(validation),
                "selection_size": len(selection),
                "validation_rmse": validation_rmse,
                "selection_rmse": float(np.sqrt(np.mean((prediction[selection] - reference[selection]) ** 2))),
                "search_budget": SEARCH_BUDGET,
                **gaps,
                **theorem,
                "old_bound_over_R": gaps["n_2eps_over_R"],
                "one_sided_bound_over_R": (
                    theorem["one_sided_bound_size"] / gaps["witness_rank_R"]
                    if gaps["witness_rank_R"] is not None else None
                ),
            })
            proxy_order = ranked_order(prediction, selection, candidate_ids)
            probe_order = top_order(selected, prediction, selection, candidate_ids)
            certificate_data.extend(certificate_rows(
                "av_gfp", model_name, seed, reference, selected, probe_order, "top_b",
                list(BUDGETS), float(gaps["g_verifier"]), gaps["witness_rank_R"],
            ))
            print(json.dumps({
                "seed": seed,
                "model": model_name,
                "g_verifier": gaps["g_verifier"],
                "R": gaps["witness_rank_R"],
                "top_candidate": int(proxy_order[0]),
            }), flush=True)

    metrics = pd.DataFrame(metric_rows)
    certificates = pd.DataFrame(certificate_data)
    splits = pd.DataFrame(split_rows)
    leakage = pd.DataFrame(leakage_rows)
    metrics.to_parquet(OUTPUT / "metrics.parquet", index=False)
    certificates.to_parquet(OUTPUT / "certificates.parquet", index=False)
    splits.to_parquet(OUTPUT / "splits.parquet", index=False)
    leakage.to_parquet(OUTPUT / "leakage_checks.parquet", index=False)
    summary = metrics.groupby("model", as_index=False).agg(
        instances=("seed", "size"),
        verifier_failures=("g_verifier", lambda values: int((values > TOLERANCE).sum())),
        mean_g_verifier=("g_verifier", "mean"),
        mean_g_hard=("g_hard", "mean"),
        mean_g_search=("g_search", "mean"),
        mean_g_total=("g_total", "mean"),
        median_R=("witness_rank_R", "median"),
        maximum_R=("witness_rank_R", "max"),
        median_old_bound_over_R=("old_bound_over_R", "median"),
        median_one_sided_bound_over_R=("one_sided_bound_over_R", "median"),
    )
    recall = certificates[certificates.genuine_failure].groupby(
        ["model", "budget"], as_index=False
    ).agg(failure_instances=("seed", "size"), top_b_recall=("certified_failure", "mean"),
          mean_tightness=("l_b_over_g_verifier", "mean"))
    summary.to_parquet(OUTPUT / "summary.parquet", index=False)
    recall.to_parquet(OUTPUT / "certificate_summary.parquet", index=False)
    config = {
        "dataset": "GFP_AEQVI_Sarkisyan_2016",
        "seeds": list(SEEDS),
        "models": ["additive_ridge", "extra_trees"],
        "budgets": list(BUDGETS),
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "selection_size": len(frame) - TRAIN_SIZE - VALIDATION_SIZE,
        "search_budget": SEARCH_BUDGET,
        "ranking": "descending proxy score; ascending lexicographic-sequence candidate ID tie-break",
        "split": "seeded permutation, disjoint train/validation/selection; exact IDs persisted",
        "trusted_score": dataset_audit["trusted_score"],
        "source_doi": dataset_audit["source_doi"],
        "model_details": model_details,
        "one_sided_set": "literal strict prediction > prediction(x_sel) - (eps_minus + e_sel), union x_sel",
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (OUTPUT / "results.md").write_text("\n".join([
        "# Replacement measured landscape: avGFP", "", "## Setup", "",
        "The declared finite candidate set is the 51,714 unique measured avGFP amino-acid "
        "sequences in the pinned ProteinGym processing of Sarkisyan et al. (2016). The trusted "
        "score is directionality-normalized `DMS_score`, sourced from mean median fluorescence "
        "brightness per amino-acid sequence. Ten persisted splits use 4,096 training, 2,048 "
        "validation, and 45,570 selection candidates.", "",
        "Models are an additive mutation-effect ridge baseline and ExtraTrees on complete "
        "positional residue tokens. Nominations are exact maxima over the declared selection set.", "",
        "## Main results", "", markdown_table(summary), "", "Top-B certification:", "",
        markdown_table(recall), "", "## Key numbers", "",
        f"Verifier failure occurred in {int((metrics.g_verifier > TOLERANCE).sum())}/{len(metrics)} "
        "model/split instances. All one-sided and global theorem containment checks passed. "
        f"All {len(leakage)} persisted split checks passed.", "", "## Negative or unexpected results", "",
        "The ProteinGym table is a finite collection of measured variants, not an exhaustive "
        "enumeration of all GFP sequences. Exact claims are relative to its declared held-out "
        "measured selection pools. The ExtraTrees token encoding is a task baseline, not a "
        "pretrained protein representation.", "", "## Files", "",
        "- `metrics.parquet`: split/model audit rows.",
        "- `certificates.parquet` and `certificate_summary.parquet`: Top-B results.",
        "- `splits.parquet` and `leakage_checks.parquet`: exact candidate identities and checks.",
        "- `raw/predictions/`: full finite predictions.",
        "- `config.json`: exact configuration and model choices.",
        "- `run.log`: command, environment, and runtime.", "",
    ]))
    (OUTPUT / "run.log").write_text("\n".join([
        f"finished_utc={datetime.now(timezone.utc).isoformat()}",
        "command=python -m experiments.audits.av_gfp",
        f"python={sys.version.replace(chr(10), ' ')}",
        f"platform={platform.platform()}",
        f"numpy={np.__version__}",
        f"pandas={pd.__version__}",
        f"scikit_learn={sklearn.__version__}",
        f"candidate_rows={len(frame)}",
        f"metric_rows={len(metrics)}",
        f"certificate_rows={len(certificates)}",
        f"split_rows={len(splits)}",
        f"elapsed_seconds={time.monotonic() - started:.6f}",
        "failed_command=Initial fit was interrupted after detecting that float64 predictions "
        "were being stored as float32; all incompatible caches were refitted and final predictions "
        "are retained exactly as float64.",
        "status=completed", "",
    ]))
    print(json.dumps({
        "metrics": len(metrics),
        "certificates": len(certificates),
        "elapsed_seconds": time.monotonic() - started,
    }))


if __name__ == "__main__":
    main()

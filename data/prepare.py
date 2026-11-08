"""Download the measured landscapes and build their candidate tables: python -m data.prepare"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DNA = np.asarray(list("ACGT"))
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
DESIGN_BENCH = "https://huggingface.co/datasets/beckhamc/design_bench_data/resolve/79006c5105c6d0da75b2d9b599f31a5e74136778"
PROTEINGYM = "https://huggingface.co/datasets/OATML-Markslab/ProteinGym_v0.1/resolve/4075aa679683f3071d527283819637f3446ca488"
TRPB = "https://data.caltech.edu/api/records/h5rah-5z170/files/data.zip/content"


def base_ids(tokens: np.ndarray, alphabet_size: int) -> np.ndarray:
    result = np.zeros(len(tokens), dtype=np.int64)
    for column in tokens.T:
        result = result * alphabet_size + column
    return result


def tokens_from_ids(ids: np.ndarray, length: int, alphabet_size: int) -> np.ndarray:
    remaining = ids.astype(np.int64).copy()
    result = np.empty((len(ids), length), dtype=np.int8)
    for position in range(length - 1, -1, -1):
        result[:, position] = remaining % alphabet_size
        remaining //= alphabet_size
    return result


def write_candidates(directory: Path, tokens: np.ndarray, scores: np.ndarray, alphabet: np.ndarray) -> None:
    candidate_id = base_ids(tokens, len(alphabet))
    order = np.argsort(candidate_id, kind="stable")
    tokens = tokens[order]
    scores = scores[order]
    candidate_id = candidate_id[order]
    sequences = np.asarray(["".join(alphabet[row]) for row in tokens])
    frame: dict[str, object] = {
        "candidate_id": candidate_id,
        "sequence": sequences,
        "f": scores.astype(np.float64),
    }
    for position in range(tokens.shape[1]):
        frame[f"token_{position}"] = tokens[:, position]
    pd.DataFrame(frame).to_parquet(directory / "candidates.parquet", index=False)


def prepare_tf_bind_8() -> dict[str, object]:
    directory = DATA / "tf_bind_8"
    x_path = directory / "raw" / "tf_bind_8-x-0.npy"
    y_path = directory / "raw" / "tf_bind_8-y-0.npy"
    x = np.load(x_path).astype(np.int8)
    y = np.load(y_path).reshape(-1).astype(float)
    ids = base_ids(x, 4)
    expected = 4**8
    counts = np.bincount(ids, minlength=expected)
    minimum = np.full(expected, np.inf)
    maximum = np.full(expected, -np.inf)
    np.minimum.at(minimum, ids, y)
    np.maximum.at(maximum, ids, y)
    # Match the exact-oracle dictionary semantics. The duplicate labels agree,
    # but the explicit reverse-unique operation makes the convention testable.
    reverse_ids = ids[::-1]
    unique_ids, first_reverse = np.unique(reverse_ids, return_index=True)
    scores = np.full(expected, np.nan)
    scores[unique_ids] = y[::-1][first_reverse]
    if not np.isfinite(scores).all():
        raise AssertionError("TF-Bind-8 does not cover all 4^8 candidates")
    tokens = tokens_from_ids(np.arange(expected), 8, 4)
    write_candidates(directory, tokens, scores, DNA)
    audit = {
        "dataset": "tf_bind_8",
        "raw_rows": int(len(x)),
        "unique_candidates": int(len(unique_ids)),
        "expected_candidates": expected,
        "duplicate_rows": int(len(x) - len(unique_ids)),
        "candidates_with_duplicates": int(np.count_nonzero(counts > 1)),
        "duplicate_candidates_with_label_disagreement": int(np.count_nonzero((counts > 1) & (maximum > minimum))),
        "maximum_duplicate_label_range": float(np.max((maximum - minimum)[counts > 1])),
        "deduplication": "last occurrence in official shard order, matching Design-Bench exact oracle",
    }
    (directory / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def prepare_tf_bind_10() -> dict[str, object]:
    directory = DATA / "tf_bind_10"
    expected = 4**10
    scores = np.full(expected, np.nan, dtype=np.float64)
    minimum = np.full(expected, np.inf, dtype=np.float32)
    maximum = np.full(expected, -np.inf, dtype=np.float32)
    counts = np.zeros(expected, dtype=np.uint16)
    raw_rows = 0
    shards: list[dict[str, object]] = []
    for shard in range(84):
        x_path = directory / "raw" / f"tf_bind_10-x-{shard}.npy"
        y_path = directory / "raw" / f"tf_bind_10-y-{shard}.npy"
        x = np.load(x_path).astype(np.int8)
        y = np.load(y_path).reshape(-1).astype(np.float32)
        if len(x) != len(y):
            raise AssertionError(f"TF-Bind-10 shard {shard} has mismatched x/y lengths")
        ids = base_ids(x, 4)
        np.add.at(counts, ids, 1)
        np.minimum.at(minimum, ids, y)
        np.maximum.at(maximum, ids, y)
        # Last row in this shard wins; later shards subsequently overwrite it.
        reverse_ids = ids[::-1]
        unique_ids, first_reverse = np.unique(reverse_ids, return_index=True)
        scores[unique_ids] = y[::-1][first_reverse]
        raw_rows += len(x)
        shards.append({"shard": shard, "rows": int(len(x))})
    if not np.isfinite(scores).all() or np.count_nonzero(counts) != expected:
        raise AssertionError("TF-Bind-10 does not cover all 4^10 candidates")
    tokens = tokens_from_ids(np.arange(expected), 10, 4)
    write_candidates(directory, tokens, scores, DNA)
    disagreements = (counts > 1) & (maximum > minimum)
    audit = {
        "dataset": "tf_bind_10",
        "raw_rows": int(raw_rows),
        "unique_candidates": expected,
        "expected_candidates": expected,
        "duplicate_rows": int(raw_rows - expected),
        "minimum_occurrences_per_candidate": int(counts.min()),
        "maximum_occurrences_per_candidate": int(counts.max()),
        "duplicate_candidates_with_label_disagreement": int(np.count_nonzero(disagreements)),
        "maximum_duplicate_label_range": float(np.max((maximum - minimum)[disagreements])),
        "deduplication": "last occurrence in numeric official shard order and row order, matching Design-Bench exact oracle",
        "shards": shards,
    }
    (directory / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def prepare_trpb() -> dict[str, object]:
    directory = DATA / "trpb"
    source = directory / "raw" / "four-site_simplified_AA_data.csv"
    raw = pd.read_csv(source)
    numeric = pd.to_numeric(raw["fitness"], errors="coerce")
    sequence = raw["AAs"].astype(str)
    valid_sequence = sequence.str.fullmatch(f"[{''.join(AMINO_ACIDS)}]{{4}}")
    valid = valid_sequence & numeric.notna() & np.isfinite(numeric)
    usable = raw.loc[valid, ["AAs"]].copy()
    usable["fitness"] = numeric[valid].astype(float)
    duplicate_rows = int(usable.AAs.duplicated(keep=False).sum())
    usable = usable.drop_duplicates("AAs", keep="first")
    alphabet = np.asarray(AMINO_ACIDS)
    lookup = {letter: index for index, letter in enumerate(alphabet)}
    tokens = np.asarray([[lookup[letter] for letter in value] for value in usable.AAs], dtype=np.int8)
    write_candidates(directory, tokens, usable.fitness.to_numpy(float), alphabet)
    audit = {
        "dataset": "trpb",
        "raw_rows": int(len(raw)),
        "valid_measured_non_stop_rows": int(valid.sum()),
        "unique_measured_candidates": int(len(usable)),
        "possible_20_pow_4_candidates": 20**4,
        "missing_unmeasured_candidates": int(20**4 - len(usable)),
        "stop_containing_rows_excluded": int((~valid_sequence).sum()),
        "nonfinite_fitness_excluded": int((~numeric.notna() | ~np.isfinite(numeric.fillna(0))).sum()),
        "duplicate_valid_rows": duplicate_rows,
        "fitness_min": float(usable.fitness.min()),
        "fitness_max": float(usable.fitness.max()),
        "imputation": "none; the separate KNN-imputed 160,000-candidate table was not used",
    }
    (directory / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


@dataclass(frozen=True)
class GB1Data:
    variants: np.ndarray
    encoded: np.ndarray
    raw_fitness: np.ndarray
    score: np.ndarray


def load_and_audit(path: Path) -> tuple[GB1Data, dict[str, object]]:
    rows = _read_xlsx(path)
    header = rows[0]
    expected = ["Variants", "HD", "Count input", "Count selected", "Fitness"]
    if header != expected:
        raise ValueError(f"unexpected GB1 columns: {header}")
    frame = pd.DataFrame(rows[1:], columns=header)
    numeric = pd.to_numeric(frame.Fitness, errors="coerce")
    variants = frame.Variants.astype(str)
    valid_variant = variants.str.fullmatch(f"[{''.join(AMINO_ACIDS)}]{{4}}")
    valid = numeric.notna() & np.isfinite(numeric) & (numeric >= 0) & valid_variant
    usable = frame.loc[valid].copy()
    usable["Fitness"] = numeric[valid].astype(float)
    duplicate_rows = int(usable.Variants.duplicated(keep=False).sum())
    usable = usable.drop_duplicates("Variants", keep="first")
    alphabet = {letter: index for index, letter in enumerate(AMINO_ACIDS)}
    encoded = np.array([[alphabet[letter] for letter in variant] for variant in usable.Variants], dtype=np.int8)
    fitness = usable.Fitness.to_numpy(float)
    data = GB1Data(usable.Variants.to_numpy(str), encoded, fitness, np.log1p(fitness))
    audit = {
        "source_doi": "10.7554/eLife.16965.024",
        "source_file": str(path),
        "source_rows_excluding_header": len(frame),
        "missing_or_nonfinite_fitness": int(numeric.isna().sum() + (~np.isfinite(numeric.fillna(0))).sum()),
        "negative_fitness": int((numeric < 0).sum()),
        "invalid_variant_strings": int((~valid_variant).sum()),
        "duplicate_rows": duplicate_rows,
        "usable_unique_measurements": len(data.variants),
        "possible_20_pow_4_variants": 20 ** 4,
        "measured_fraction": len(data.variants) / 20 ** 4,
        "fitness_min": float(fitness.min()),
        "fitness_max": float(fitness.max()),
        "score_transformation": "log1p(nonnegative experimental fitness); monotone, so optima are unchanged",
        "filtering": "finite nonnegative fitness and four standard amino acids; duplicates keep first after audit",
    }
    return data, audit


def _read_xlsx(path: Path) -> list[list[object]]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as archive:
        root = ET.parse(archive.open("xl/sharedStrings.xml")).getroot()
        strings = ["".join(node.text or "" for node in item.iter(namespace + "t")) for item in root]
        result: list[list[object]] = []
        for _, row in ET.iterparse(archive.open("xl/worksheets/sheet1.xml"), events=("end",)):
            if row.tag != namespace + "row":
                continue
            values: list[object] = []
            for cell in row.findall(namespace + "c"):
                value = cell.find(namespace + "v")
                raw = "" if value is None else value.text or ""
                values.append(strings[int(raw)] if cell.get("t") == "s" else float(raw))
            result.append(values)
            row.clear()
    return result


def prepare_gb1() -> dict[str, object]:
    directory = DATA / "gb1"
    source = directory / "elife-16965-supp1-v4.xlsx"
    data, source_audit = load_and_audit(source)
    alphabet = np.asarray(AMINO_ACIDS)
    write_candidates(directory, data.encoded, data.score, alphabet)
    audit = {
        **source_audit,
        "dataset": "gb1",
        "selection_claim_scope": "measured candidates only; no imputed variants",
    }
    audit["source_file"] = str(source.relative_to(ROOT))
    (directory / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


def download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.rename(destination)


def fetch(name: str) -> None:
    raw = DATA / name / "raw"
    if name == "tf_bind_8":
        for axis in ("x", "y"):
            download(f"{DESIGN_BENCH}/tf_bind_8-SIX6_REF_R1/tf_bind_8-{axis}-0.npy", raw / f"tf_bind_8-{axis}-0.npy")
    elif name == "tf_bind_10":
        for shard in range(84):
            for axis in ("x", "y"):
                download(f"{DESIGN_BENCH}/tf_bind_10-pho4/tf_bind_10-{axis}-{shard}.npy", raw / f"tf_bind_10-{axis}-{shard}.npy")
    elif name == "trpb":
        download(TRPB, raw / "data.zip")
        table = raw / "four-site_simplified_AA_data.csv"
        if not table.exists():
            with ZipFile(raw / "data.zip") as archive:
                member = next(item for item in archive.namelist() if item.endswith(table.name))
                table.write_bytes(archive.read(member))
    elif name == "av_gfp":
        download(f"{PROTEINGYM}/ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv?download=true", raw / "GFP_AEQVI_Sarkisyan_2016.csv")
        download(f"{PROTEINGYM}/ProteinGym_reference_file_substitutions.csv?download=true", raw / "ProteinGym_reference_file_substitutions.csv")


def prepare_av_gfp() -> dict[str, object]:
    from experiments.audits.av_gfp import prepare

    return prepare()[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["tf_bind_8", "trpb", "gb1", "av_gfp"])
    args = parser.parse_args()
    functions = {
        "tf_bind_8": prepare_tf_bind_8,
        "tf_bind_10": prepare_tf_bind_10,
        "trpb": prepare_trpb,
        "gb1": prepare_gb1,
        "av_gfp": prepare_av_gfp,
    }
    for name in args.datasets:
        fetch(name)
        audit = functions[name]()
        print(json.dumps({key: value for key, value in audit.items() if key != "shards"}, sort_keys=True))


if __name__ == "__main__":
    main()

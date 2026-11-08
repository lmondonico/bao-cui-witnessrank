# Datasets

The four landscapes are TF-Bind-8, TrpB, GB1, and avGFP. Preparation writes a cleaning audit
(`dataset_audit.json`) and the processed candidate table into each subdirectory. TF-Bind-10 is
documented only to explain why it is excluded.

Run from the repository root:

```bash
.venv/bin/python -m data.prepare
# Optional, to inspect the TF-Bind-10 exclusion:
.venv/bin/python -m data.prepare tf_bind_10
```

The script downloads the sources at pinned revisions and builds the candidate tables. Raw downloads
and processed tables are excluded from Git; the GB1 source workbook is tracked. The sections below
record sources, candidate definitions, and duplicate handling.

## TF-Bind-8 provenance and preprocessing

Artifact paths in this section are relative to `data/tf_bind_8/`.

- Scientific source: Barrera et al. (2016), *Science*, DOI `10.1126/science.aad2257`.
- Benchmark source: Trabucco et al. (2022), Design-Bench, PMLR 162; task `TFBind8-Exact-v0`, transcription factor `SIX6_REF_R1`.
- Code used to define the source shard and exact-oracle behavior: `.external/design-bench` (MIT license).
- Duplicate audit: `dataset_audit.json`.
- Processed finite table: `candidates.parquet`, using lexicographic `ACGT` integer IDs.

The 65,792 raw rows contain all 65,536 length-eight sequences and 256 duplicate rows. Duplicate labels agree exactly. The preprocessing retains the last occurrence, matching the benchmark exact oracle's dictionary-assignment behavior.

## TrpB provenance and preprocessing

Artifact paths in this section are relative to `data/trpb/`.

- Source: Johnston et al. (2024), *PNAS*, DOI `10.1073/pnas.2400439121`.
- Author deposit: CaltechDATA DOI `10.22002/h5rah-5z170`, version 2, updated 2024-06-24.
- Deposit license: CC0 1.0 according to the CaltechDATA record metadata.
- Downloaded archive: `raw/data.zip`, matching the deposit metadata.
- Measured source table: `raw/four-site_simplified_AA_data.csv`.
- Processed finite table: `candidates.parquet`, using lexicographic `ACDEFGHIKLMNPQRSTVWY` integer IDs.

Only the 159,129 directly quantified, non-stop four-amino-acid variants are used. Stop-containing rows are excluded. The 871 unmeasured variants remain absent; the separately deposited KNN-imputed 160,000-candidate table is not used.

## GB1 provenance and preprocessing

Artifact paths in this section are relative to `data/gb1/`.

- Source: Wu et al. (2016), *eLife*, DOI `10.7554/eLife.16965.024`, Supplementary file 1.
- The exact source workbook is retained at `data/gb1/elife-16965-supp1-v4.xlsx`.
- Processed finite table: `candidates.parquet`, using `log1p` of nonnegative fitness and lexicographic `ACDEFGHIKLMNPQRSTVWY` integer IDs.

The audit retains 149,361 measured unique variants. No imputed variant is used. Every optimum and regret claim is relative only to the declared measured selection pool.

## avGFP measured landscape

Artifact paths in this section are relative to `data/av_gfp/`.

`data/av_gfp/` contains the ProteinGym processing of the Sarkisyan et al. avGFP fluorescence
landscape.

### Provenance

- Primary study: Sarkisyan et al., *Local fitness landscape of the green fluorescent protein*,
  Nature 533, 397–401 (2016), DOI `10.1038/nature17995`.
- Curated source: ProteinGym v0.1 (OATML/Marks Lab).
- Files: `ProteinGym_substitutions/GFP_AEQVI_Sarkisyan_2016.csv` and
  `ProteinGym_reference_file_substitutions.csv`.

The trusted value is ProteinGym `DMS_score`. The reference metadata identifies its raw phenotype
as `mean_medianBrightness_per_aaseq` with directionality `+1`; higher is therefore better.

### Candidate and quality audit

The declared table contains 51,714 measured amino-acid variants. Mutation strings are applied to
the pinned 238-residue target sequence and independently validated against the reference residue.
The preprocessing requires unique mutation strings, unique reconstructed sequences, finite scores,
no missing values, and no conflicting sequence labels. Exact results are saved in
`dataset_audit.json`.

This is a finite measured candidate collection rather than an exhaustive enumeration of protein
sequence space. All optimization and certification claims are relative to the persisted measured
selection pools.

## TF-Bind-10 provenance and preprocessing

Artifact paths in this section are relative to `data/tf_bind_10/`.

- Scientific source reported by Design-Bench: Barrera et al. (2016), *Science*, DOI `10.1126/science.aad2257`.
- Benchmark source: Trabucco et al. (2022), Design-Bench, PMLR 162; task `TFBind10-Exact-v0`, transcription factor `pho4`.
- Code used to define the source shards and exact-oracle behavior: `.external/design-bench` (MIT license).
- Per-shard row counts and duplicate audit: `dataset_audit.json`.
- Processed finite table: `candidates.parquet`, using lexicographic `ACGT` integer IDs.

### Material source limitation

The 84 official shards contain 8,321,066 rows covering all 1,048,576 length-ten sequences 5 to 8 times. Every sequence has at least two disagreeing labels; the maximum within-sequence range is 3.9872336. This was independently inspected before the audit.

Design-Bench's `TFBind10Oracle.__init__` iterates all internal samples and performs `sequence_to_score[tuple(x)] = y`, so later occurrences overwrite earlier ones. The audit reproduces this exact last-occurrence behavior using numeric shard order 0 through 83 and row order. Consequently, the resulting finite landscape is source-order dependent. No aggregation was invented, and this limitation is reported prominently in the results.

Design-Bench code is MIT-licensed; this does not establish a license for the data.

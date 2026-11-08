# Hard to Find, Easy to Fool: Witness Rank for Budgeted Scientific Verification

**Paper:** [![Paper](https://img.shields.io/badge/Paper-Witness%20Rank-blue)](https://lucamondonico.com/research/witness-rank/)

Code for **“Hard to Find, Easy to Fool: Witness Rank for Budgeted Scientific Verification”**
(AI for Science workshop, NeurIPS 2026).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m data.prepare
```

## Reproduction

Run the scripts in `experiments/` as modules from the repository root, for example
`.venv/bin/python -m experiments.witness_rank.margins`. The RaM and COM runs also need the authors'
code ([Offline-RaM](https://github.com/lamda-bbo/Offline-RaM),
[design-baselines](https://github.com/brandontrabucco/design-baselines)) in `.external/`. Figures
and their source data are in `figures/`.

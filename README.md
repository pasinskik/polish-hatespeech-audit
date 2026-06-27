# polish-hatespeech-audit

This repository contains a lightweight notebook-first setup for auditing Polish hate-speech models on a binary harmful vs non-harmful task.

## Project layout

```text
polish-hatespeech-audit/
├── notebooks/
│   └── 01_baseline_ptaszynski_hatecheck.ipynb
├── data/
│   └── test.csv
├── results/
│   └── .gitkeep
├── src/
│   ├── inference.py
│   ├── metrics.py
│   └── counterfactuals.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebook goal

`notebooks/01_baseline_ptaszynski_hatecheck.ipynb` builds a baseline audit using:
- `ptaszynski/bert-base-polish-cyberbullying`
- binary label collapse: non-harmful (`0`) vs harmful (`1`, `2`)
- basic functional-category metrics and a simple name-swap counterfactual sensitivity check.

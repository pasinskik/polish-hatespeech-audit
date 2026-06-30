"""Inferencja na pełnym counterfactual datasecie.

Czyta data/counterfactual_dataset.parquet (45 780 wierszy), uruchamia model,
zapisuje predict_proba per (case, variant) jako parquet + log JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import BATCH_SIZE, DATA_DIR, MAX_LENGTH, PTASZYNSKI_MODEL_ID, TRELBERT_FINETUNED_MODEL_ID, RESULTS_DIR, SEED
from inference import HarmfulClassifier

MODELS = {
    "ptaszynski": dict(model_id=PTASZYNSKI_MODEL_ID, harmful_index=1),
    "trelbert": dict(model_id=TRELBERT_FINETUNED_MODEL_ID, harmful_index=1),
}

LABEL_GOLD_TO_INT = {"hateful": 1, "non-hateful": 0}


def run(model_key: str, threshold: float = 0.5) -> None:
    cfg = MODELS[model_key]
    print(f"Model: {cfg['model_id']} (harmful_index={cfg['harmful_index']})")

    df = pd.read_parquet(DATA_DIR / "counterfactual_dataset.parquet")
    df["y_true"] = df["label_gold"].map(LABEL_GOLD_TO_INT).astype(int)
    print(f"Wierszy: {len(df)}  ·  cases: {df['mhc_case_id'].nunique()}  ·  variants: {df['variant'].nunique()}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clf = HarmfulClassifier(
        model_id=cfg["model_id"],
        harmful_index=cfg["harmful_index"],
        max_length=MAX_LENGTH,
    )
    print(f"Device: {clf.device}")

    probs = clf.predict_proba(df["text"].tolist(), batch_size=BATCH_SIZE)
    df["p_harmful"] = probs
    df["y_pred"] = (df["p_harmful"] >= threshold).astype(int)

    out_cols = [
        "mhc_case_id",
        "functionality",
        "label_gold",
        "target_ident",
        "prefix_class",
        "prefix_name",
        "variant",
        "y_true",
        "p_harmful",
        "y_pred",
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"cf_{model_key}_predictions.parquet"
    df[out_cols].to_parquet(pred_path, index=False)
    print(f"\nZapisano predykcje: {pred_path}")

    print("\n=== ŚREDNIE P(harmful) per prefix_class ===")
    print(df.groupby("prefix_class")["p_harmful"].agg(["mean", "median", "std", "count"]).round(4).to_string())

    print("\n=== ŚREDNIE P(harmful) per variant (sortowane) ===")
    print(df.groupby("variant")["p_harmful"].agg(["mean", "median"]).round(4).sort_values("mean").to_string())

    pivot = df.pivot_table(index="mhc_case_id", columns="variant", values="p_harmful")
    print("\n=== Δp = mean(p_variant - p_original), preview ===")
    deltas = {var: float((pivot[var] - pivot["original"]).mean()) for var in pivot.columns if var != "original"}
    print(pd.Series(deltas).sort_values().round(4).to_string())

    try:
        model_info = HfApi().model_info(cfg["model_id"])
        model_rev = model_info.sha
    except Exception:
        model_rev = None

    run_log = {
        "model_key": model_key,
        "model_id": cfg["model_id"],
        "model_revision": model_rev,
        "harmful_index": cfg["harmful_index"],
        "threshold": threshold,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "device": str(clf.device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dataset_path": str(DATA_DIR / "counterfactual_dataset.parquet"),
        "n_rows": int(len(df)),
        "n_cases": int(df["mhc_case_id"].nunique()),
        "n_variants": int(df["variant"].nunique()),
        "mean_p_per_prefix_class": {k: float(v) for k, v in df.groupby("prefix_class")["p_harmful"].mean().items()},
    }
    log_path = RESULTS_DIR / f"cf_{model_key}_run.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nZapisano log runu: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ptaszynski", choices=list(MODELS))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.model, args.threshold)


if __name__ == "__main__":
    main()

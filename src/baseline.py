"""Baseline: inference + metryki dla modelu na oryginalnym polskim HateCheck."""
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
from config import BATCH_SIZE, DATA_DIR, MAX_LENGTH, PTASZYNSKI_MODEL_ID, RESULTS_DIR, SEED
from inference import HarmfulClassifier
from metrics import metrics_overall, metrics_per_functionality, metrics_per_target_ident

MODELS = {
    "ptaszynski": dict(model_id=PTASZYNSKI_MODEL_ID, harmful_index=1),
}

LABEL_GOLD_TO_INT = {"hateful": 1, "non-hateful": 0}


def run(model_key: str, threshold: float = 0.5) -> None:
    cfg = MODELS[model_key]
    print(f"Model: {cfg['model_id']} (harmful_index={cfg['harmful_index']})")

    df = pd.read_parquet(DATA_DIR / "hatecheck_pl.parquet")
    df = df[df["label_gold"].isin(LABEL_GOLD_TO_INT)].reset_index(drop=True)
    df["y_true"] = df["label_gold"].map(LABEL_GOLD_TO_INT).astype(int)
    print(f"Przykładów: {len(df)} (hateful={int((df.y_true==1).sum())}, non-hateful={int((df.y_true==0).sum())})")

    clf = HarmfulClassifier(model_id=cfg["model_id"], harmful_index=cfg["harmful_index"], max_length=MAX_LENGTH)
    print(f"Device: {clf.device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    probs = clf.predict_proba(df["test_case"].tolist(), batch_size=BATCH_SIZE)
    df["p_harmful"] = probs
    df["y_pred"] = (df["p_harmful"] >= threshold).astype(int)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"baseline_{model_key}_predictions.parquet"
    df[["mhc_case_id", "functionality", "target_ident", "label_gold", "y_true", "p_harmful", "y_pred"]].to_parquet(pred_path, index=False)
    print(f"\nZapisano predykcje: {pred_path}")

    overall = metrics_overall(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df["p_harmful"].to_numpy())
    print("\n=== METRYKI OGÓLNE ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    per_func = metrics_per_functionality(df)
    func_path = RESULTS_DIR / f"baseline_{model_key}_per_functionality.parquet"
    per_func.reset_index().to_parquet(func_path, index=False)
    print(f"\n=== METRYKI PER FUNCTIONALITY ===")
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 140)
    print(per_func.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nZapisano per-functionality: {func_path}")

    per_ident = metrics_per_target_ident(df)
    ident_path = RESULTS_DIR / f"baseline_{model_key}_per_target_ident.parquet"
    per_ident.reset_index().to_parquet(ident_path, index=False)
    print(f"\n=== METRYKI PER TARGET_IDENT (grupa chroniona) ===")
    print(per_ident.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nZapisano per-target_ident: {ident_path}")

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
        "dataset_path": str(DATA_DIR / "hatecheck_pl.parquet"),
        "n_examples": int(len(df)),
        "overall_metrics": overall,
    }
    log_path = RESULTS_DIR / f"baseline_{model_key}_run.json"
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

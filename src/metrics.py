"""Metryki ewaluacyjne dla binarnej klasyfikacji harmful/non-harmful."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def fpr_fnr(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    neg = y_true == 0
    pos = y_true == 1
    fpr = float((y_pred[neg] == 1).mean()) if neg.any() else float("nan")
    fnr = float((y_pred[pos] == 0).mean()) if pos.any() else float("nan")
    return fpr, fnr


def metrics_overall(y_true: np.ndarray, y_pred: np.ndarray, p_harmful: np.ndarray | None = None) -> dict[str, float]:
    fpr, fnr = fpr_fnr(y_true, y_pred)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": fpr,
        "fnr": fnr,
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
    }
    if p_harmful is not None and (y_true == 0).any() and (y_true == 1).any():
        out["roc_auc"] = float(roc_auc_score(y_true, p_harmful))
    else:
        out["roc_auc"] = float("nan")
    return out


def metrics_per_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """`df` musi mieć kolumny: <group_col>, y_true, y_pred, p_harmful."""
    rows = []
    for key, grp in df.groupby(group_col, dropna=False):
        m = metrics_overall(
            grp["y_true"].to_numpy(),
            grp["y_pred"].to_numpy(),
            grp["p_harmful"].to_numpy(),
        )
        m[group_col] = key
        rows.append(m)
    out = pd.DataFrame(rows).set_index(group_col).sort_index()
    cols = ["n", "n_pos", "n_neg", "accuracy", "macro_f1", "fpr", "fnr", "roc_auc"]
    return out[cols]


def metrics_per_functionality(df: pd.DataFrame) -> pd.DataFrame:
    return metrics_per_group(df, "functionality")


def metrics_per_target_ident(df: pd.DataFrame) -> pd.DataFrame:
    return metrics_per_group(df, "target_ident")

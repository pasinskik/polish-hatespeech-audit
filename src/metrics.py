"""Evaluation metrics for binary hate-speech classification."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def classification_summary(y_true: Sequence[int], y_pred: Sequence[int]) -> str:
    return classification_report(y_true, y_pred, target_names=["non-harmful", "harmful"])


def confusion_matrix_df(y_true: Sequence[int], y_pred: Sequence[int]) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        matrix,
        index=["true_non_harmful", "true_harmful"],
        columns=["pred_non_harmful", "pred_harmful"],
    )

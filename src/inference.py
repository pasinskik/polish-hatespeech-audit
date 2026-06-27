"""Inference helpers for Polish hate-speech experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from transformers import pipeline

DEFAULT_MODEL_NAME = "ptaszynski/bert-base-polish-cyberbullying"


@dataclass(frozen=True)
class BinaryPrediction:
    text: str
    model_label: str
    score: float
    harmful_pred: int


def _to_binary_label(raw_label: str) -> int:
    label = str(raw_label)
    if label.isdigit():
        return int(label) > 0
    if label.startswith("LABEL_"):
        suffix = label.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix) > 0
    lowered = label.lower()
    if lowered in {"non-harmful", "non_harmful", "neutral", "none"}:
        return 0
    return 1


def predict_binary(
    texts: Sequence[str] | Iterable[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Run model inference and collapse labels to binary: harmful vs non-harmful."""
    texts = list(texts)
    clf = pipeline("text-classification", model=model_name, tokenizer=model_name)
    rows: list[BinaryPrediction] = []
    for text, prediction in zip(
        texts,
        clf(texts, truncation=True, max_length=256, batch_size=batch_size),
    ):
        rows.append(
            BinaryPrediction(
                text=text,
                model_label=prediction["label"],
                score=float(prediction["score"]),
                harmful_pred=int(_to_binary_label(prediction["label"])),
            )
        )
    return pd.DataFrame(rows)

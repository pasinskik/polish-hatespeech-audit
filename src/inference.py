"""Inferencja batchowa dla modeli klasyfikujących polski hate speech."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class HarmfulClassifier:
    """Wrapper na binarny model harmful/non-harmful.

    `harmful_index` to indeks klasy oznaczającej harmful w głowicy modelu.
    Dla `ptaszynski/bert-base-polish-cyberbullying` ustalone empirycznie = 1.
    """

    model_id: str
    harmful_index: int
    max_length: int = 256

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.device = pick_device()
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_proba(self, texts: Iterable[str], batch_size: int = 32) -> np.ndarray:
        texts = list(texts)
        out = np.empty(len(texts), dtype=np.float32)
        for i in tqdm(range(0, len(texts), batch_size), desc=f"infer:{self.model_id.split('/')[-1]}"):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, self.harmful_index]
            out[i : i + len(batch)] = probs.cpu().numpy()
        return out

"""Buduje pełny counterfactual dataset (long format) na bazie polskiego HateCheck.

Dla każdego z N przykładów generowane jest 12 wariantów tekstu:
  - original (bez prefixu)
  - neutral  (Osoba: x)
  - feminine (5 imion żeńskich)
  - masculine (5 imion męskich)

Wynik: data/counterfactual_dataset.parquet (long format, ~46k wierszy).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    FEMININE_NAMES,
    MASCULINE_NAMES,
    NEUTRAL_PREFIX,
    PREFIX_FORMAT,
)


def build_variants(test_case: str) -> list[dict]:
    """Zwraca listę 12 słowników (prefix_class, prefix_name, variant, text) dla jednego przykładu."""
    rows: list[dict] = [
        {"prefix_class": "original", "prefix_name": None, "variant": "original", "text": test_case},
        {
            "prefix_class": "neutral",
            "prefix_name": None,
            "variant": "neutral",
            "text": PREFIX_FORMAT.format(name=NEUTRAL_PREFIX, text=test_case),
        },
    ]
    for name in FEMININE_NAMES:
        rows.append(
            {
                "prefix_class": "feminine",
                "prefix_name": name,
                "variant": f"feminine_{name}",
                "text": PREFIX_FORMAT.format(name=name, text=test_case),
            }
        )
    for name in MASCULINE_NAMES:
        rows.append(
            {
                "prefix_class": "masculine",
                "prefix_name": name,
                "variant": f"masculine_{name}",
                "text": PREFIX_FORMAT.format(name=name, text=test_case),
            }
        )
    return rows


def main() -> None:
    src_path = DATA_DIR / "hatecheck_pl.parquet"
    out_path = DATA_DIR / "counterfactual_dataset.parquet"

    print(f"Czytam: {src_path}")
    base = pd.read_parquet(src_path)
    base = base[base["label_gold"].isin(["hateful", "non-hateful"])].reset_index(drop=True)
    print(f"Przykładów bazowych: {len(base)}")

    base["_variants"] = base["test_case"].apply(build_variants)
    long = base.explode("_variants", ignore_index=True)
    variant_df = pd.json_normalize(long["_variants"])
    long = pd.concat([long.drop(columns=["_variants"]), variant_df], axis=1)

    long = long[
        [
            "mhc_case_id",
            "functionality",
            "label_gold",
            "target_ident",
            "prefix_class",
            "prefix_name",
            "variant",
            "text",
        ]
    ]

    n_cases = base["mhc_case_id"].nunique()
    n_variants_per_case = 2 + len(FEMININE_NAMES) + len(MASCULINE_NAMES)
    expected = n_cases * n_variants_per_case

    assert len(long) == expected, f"oczekiwane {expected}, dostałem {len(long)}"
    assert (
        long.groupby("mhc_case_id").size().nunique() == 1
    ), "nie każdy case ma tyle samo wariantów"
    assert long["text"].notna().all(), "brakujące teksty"
    assert (long["text"].str.len() > 0).all(), "pusty tekst"
    assert (
        not long.duplicated(["mhc_case_id", "variant"]).any()
    ), "duplikat (case, variant)"

    print(f"\nLong format: {len(long)} wierszy = {n_cases} × {n_variants_per_case}")
    print(f"\nRozkład prefix_class:\n{long['prefix_class'].value_counts()}")
    print(f"\nRozkład wariantów (powinno być po {n_cases} każdy):")
    print(long["variant"].value_counts().sort_index().to_string())

    long.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nZapisano: {out_path} ({size_mb:.2f} MB)")

    print("\nPrzykład pełnego rekordu (polish-2656, wszystkie 12 wariantów):")
    sample = long[long["mhc_case_id"] == long["mhc_case_id"].iloc[0]] if "polish-2656" not in long["mhc_case_id"].values else long[long["mhc_case_id"] == "polish-2656"]
    print(sample[["variant", "prefix_class", "prefix_name", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()

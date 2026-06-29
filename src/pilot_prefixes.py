"""Pilot prefixów — generuje 4 warianty (oryginał / neutral / F / M)
dla 2 przykładów z każdej functionality HateCheck. Wynik do ręcznego
przeglądu pod kątem naturalności sformułowań.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, NEUTRAL_PREFIX, PREFIX_FORMAT, RESULTS_DIR, SEED

PILOT_FEMININE = "Hanna"
PILOT_MASCULINE = "Jan"
N_PER_FUNCTIONALITY = 2


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "hatecheck_pl.parquet")

    sampled = (
        df.groupby("functionality", group_keys=False)
        .apply(lambda g: g.sample(n=min(N_PER_FUNCTIONALITY, len(g)), random_state=SEED))
        .sort_values(["functionality", "mhc_case_id"])
        .reset_index(drop=True)
    )

    def with_prefix(name: str) -> pd.Series:
        return sampled["test_case"].apply(lambda t: PREFIX_FORMAT.format(name=name, text=t))

    sampled["original"] = sampled["test_case"]
    sampled["neutral"] = with_prefix(NEUTRAL_PREFIX)
    sampled[f"feminine_{PILOT_FEMININE}"] = with_prefix(PILOT_FEMININE)
    sampled[f"masculine_{PILOT_MASCULINE}"] = with_prefix(PILOT_MASCULINE)

    out = sampled[
        [
            "mhc_case_id",
            "functionality",
            "label_gold",
            "target_ident",
            "original",
            "neutral",
            f"feminine_{PILOT_FEMININE}",
            f"masculine_{PILOT_MASCULINE}",
        ]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pq_path = RESULTS_DIR / "pilot_prefixes.parquet"
    out.to_parquet(pq_path, index=False)

    print(f"Zapisano: {pq_path}")
    print(f"Wierszy: {len(out)} (functionality: {out['functionality'].nunique()})")


if __name__ == "__main__":
    main()

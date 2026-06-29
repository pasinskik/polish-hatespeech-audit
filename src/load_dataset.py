"""Pobiera polski HateCheck z HuggingFace i zapisuje jako Parquet."""
from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, HATECHECK_HF_ID


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Ładuję {HATECHECK_HF_ID}...")
    ds = load_dataset(HATECHECK_HF_ID)
    print(f"Splity: {list(ds.keys())}")
    for split, d in ds.items():
        print(f"  {split}: n={len(d)}, kolumny={d.column_names}")

    split_name = next(iter(ds.keys()))
    df = ds[split_name].to_pandas()
    print(f"\nPrzykładowe wiersze:\n{df.head(3).to_string()}")
    print(f"\nKształt: {df.shape}")
    print(f"Kolumny: {list(df.columns)}")

    if "label_gold" in df.columns:
        print(f"\nRozkład label_gold:\n{df['label_gold'].value_counts()}")
    if "functionality" in df.columns:
        print(f"\nLiczba functionality: {df['functionality'].nunique()}")
        print(f"Functionality (top 30):\n{df['functionality'].value_counts().head(30)}")
    if "target_ident" in df.columns:
        print(f"\nGrupy docelowe:\n{df['target_ident'].value_counts()}")

    out = DATA_DIR / "hatecheck_pl.parquet"
    df.to_parquet(out, index=False)
    print(f"\nZapisano: {out}")


if __name__ == "__main__":
    main()

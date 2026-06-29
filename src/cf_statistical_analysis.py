"""Analiza statystyczna eksperymentu counterfactual.

Dla każdej z 6 par porównań (np. feminine_avg vs original, feminine_avg vs neutral)
liczone są: Δp, flip rate, paired Wilcoxon (Δp), McNemar (flip rate), Cohen's h.
Wszystko trzy razy: globalnie, per functionality (27), per target_ident (8).
Korekta wielokrotnego testowania: Holm-Bonferroni w obrębie rodziny porównań.

Wyniki: parquet w results/.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FEMININE_NAMES, MASCULINE_NAMES, RESULTS_DIR

THRESHOLD = 0.5

PAIRS = [
    ("feminine_avg", "original", "fem_vs_orig", "Effect of any feminine name vs original (total feminine effect)"),
    ("masculine_avg", "original", "masc_vs_orig", "Effect of any masculine name vs original (total masculine effect)"),
    ("neutral", "original", "neutral_vs_orig", "Effect of bare prefix format (no name)"),
    ("feminine_avg", "neutral", "fem_vs_neutral", "Gender effect of feminine names net of prefix format"),
    ("masculine_avg", "neutral", "masc_vs_neutral", "Gender effect of masculine names net of prefix format"),
    ("feminine_avg", "masculine_avg", "fem_vs_masc", "Feminine vs masculine direct comparison"),
]


def load_wide(predictions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wczytuje long-format predykcje, robi wide pivot per p_harmful i y_pred,
    dodaje feminine_avg / masculine_avg per case (średnia po 5 imionach).
    Zwraca: (p_wide, y_wide, meta) gdzie meta ma functionality, target_ident, label_gold per case.
    """
    df = pd.read_parquet(predictions_path)
    df["target_ident"] = df["target_ident"].fillna("__none__")

    p_wide = df.pivot(index="mhc_case_id", columns="variant", values="p_harmful")
    y_wide = df.pivot(index="mhc_case_id", columns="variant", values="y_pred")

    fem_cols = [f"feminine_{n}" for n in FEMININE_NAMES]
    masc_cols = [f"masculine_{n}" for n in MASCULINE_NAMES]
    p_wide["feminine_avg"] = p_wide[fem_cols].mean(axis=1)
    p_wide["masculine_avg"] = p_wide[masc_cols].mean(axis=1)
    y_wide["feminine_avg"] = (p_wide["feminine_avg"] >= THRESHOLD).astype(int)
    y_wide["masculine_avg"] = (p_wide["masculine_avg"] >= THRESHOLD).astype(int)

    meta = (
        df[["mhc_case_id", "functionality", "target_ident", "label_gold"]]
        .drop_duplicates("mhc_case_id")
        .set_index("mhc_case_id")
    )
    return p_wide, y_wide, meta


@dataclass
class TestResult:
    pair: str
    pair_label: str
    slice_type: str
    slice_value: str
    n: int
    delta_mean: float
    delta_median: float
    delta_ci_low: float
    delta_ci_high: float
    flip_rate: float
    wilcoxon_stat: float
    wilcoxon_p: float
    mcnemar_stat: float
    mcnemar_p: float
    cohens_h: float


def bootstrap_ci(arr: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    if len(arr) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot[i] = sample.mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def cohens_h(p1: float, p2: float) -> float:
    p1 = np.clip(p1, 1e-9, 1 - 1e-9)
    p2 = np.clip(p2, 1e-9, 1 - 1e-9)
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def run_pair_on_slice(
    pair: tuple[str, str, str, str],
    p_wide: pd.DataFrame,
    y_wide: pd.DataFrame,
    case_ids: list[str],
    slice_type: str,
    slice_value: str,
) -> TestResult:
    var_a, var_b, code, label = pair
    p_a = p_wide.loc[case_ids, var_a].to_numpy()
    p_b = p_wide.loc[case_ids, var_b].to_numpy()
    y_a = y_wide.loc[case_ids, var_a].to_numpy()
    y_b = y_wide.loc[case_ids, var_b].to_numpy()
    delta = p_a - p_b
    n = int(len(delta))

    delta_mean = float(np.mean(delta)) if n else float("nan")
    delta_median = float(np.median(delta)) if n else float("nan")
    ci_low, ci_high = bootstrap_ci(delta) if n >= 2 else (float("nan"), float("nan"))
    flip = (y_a != y_b).astype(int)
    flip_rate = float(flip.mean()) if n else float("nan")

    if n >= 10 and np.any(delta != 0):
        try:
            w_stat, w_p = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
            w_stat, w_p = float(w_stat), float(w_p)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), float("nan")

    if n >= 10:
        b = int(np.sum((y_a == 1) & (y_b == 0)))
        c = int(np.sum((y_a == 0) & (y_b == 1)))
        if b + c >= 1:
            try:
                m = mcnemar([[0, b], [c, 0]], exact=(b + c < 25), correction=True)
                m_stat, m_p = float(m.statistic), float(m.pvalue)
            except Exception:
                m_stat, m_p = float("nan"), float("nan")
        else:
            m_stat, m_p = float("nan"), 1.0
    else:
        m_stat, m_p = float("nan"), float("nan")

    p_a_rate = float(y_a.mean()) if n else float("nan")
    p_b_rate = float(y_b.mean()) if n else float("nan")
    h = cohens_h(p_a_rate, p_b_rate) if n else float("nan")

    return TestResult(
        pair=code,
        pair_label=label,
        slice_type=slice_type,
        slice_value=slice_value,
        n=n,
        delta_mean=delta_mean,
        delta_median=delta_median,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        flip_rate=flip_rate,
        wilcoxon_stat=w_stat,
        wilcoxon_p=w_p,
        mcnemar_stat=m_stat,
        mcnemar_p=m_p,
        cohens_h=h,
    )


def add_holm_correction(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Holm-Bonferroni per grupa (np. per pair w obrębie functionality)."""
    df = df.copy()
    for col in ["wilcoxon_p", "mcnemar_p"]:
        adj_col = col.replace("_p", "_p_holm")
        rej_col = col.replace("_p", "_reject_holm")
        df[adj_col] = np.nan
        df[rej_col] = False
        for _, idx in df.groupby(group_cols).groups.items():
            sub = df.loc[idx, col].dropna()
            if len(sub) == 0:
                continue
            reject, p_adj, _, _ = multipletests(sub.values, alpha=0.05, method="holm")
            df.loc[sub.index, adj_col] = p_adj
            df.loc[sub.index, rej_col] = reject
    return df


def main() -> None:
    pred_path = RESULTS_DIR / "cf_ptaszynski_predictions.parquet"
    p_wide, y_wide, meta = load_wide(pred_path)
    all_cases = p_wide.index.tolist()
    print(f"Cases: {len(all_cases)}, variants in p_wide: {p_wide.shape[1]}")

    results: list[TestResult] = []

    for pair in PAIRS:
        results.append(run_pair_on_slice(pair, p_wide, y_wide, all_cases, "global", "all"))

    for func, cases in meta.groupby("functionality").groups.items():
        for pair in PAIRS:
            results.append(run_pair_on_slice(pair, p_wide, y_wide, cases.tolist(), "functionality", func))

    for tgt, cases in meta.groupby("target_ident").groups.items():
        for pair in PAIRS:
            results.append(run_pair_on_slice(pair, p_wide, y_wide, cases.tolist(), "target_ident", tgt))

    df = pd.DataFrame([r.__dict__ for r in results])
    df = add_holm_correction(df, group_cols=["pair", "slice_type"])

    out = RESULTS_DIR / "cf_ptaszynski_statistical_tests.parquet"
    df.to_parquet(out, index=False)
    print(f"\nZapisano: {out}  ({len(df)} testów)")

    print("\n=== GLOBAL: 6 par porównań ===")
    global_df = df[df["slice_type"] == "global"].copy()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(
        global_df[
            [
                "pair",
                "n",
                "delta_mean",
                "delta_ci_low",
                "delta_ci_high",
                "flip_rate",
                "wilcoxon_p",
                "mcnemar_p",
                "cohens_h",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )

    print("\n=== Per-functionality summary (pair=fem_vs_masc, significant after Holm) ===")
    fm = df[(df["slice_type"] == "functionality") & (df["pair"] == "fem_vs_masc")].copy()
    print(
        fm[
            [
                "slice_value",
                "n",
                "delta_mean",
                "flip_rate",
                "wilcoxon_p_holm",
                "wilcoxon_reject_holm",
                "cohens_h",
            ]
        ].sort_values("delta_mean").to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )

    print("\n=== Per-target_ident summary (pair=fem_vs_masc) ===")
    tm = df[(df["slice_type"] == "target_ident") & (df["pair"] == "fem_vs_masc")].copy()
    print(
        tm[
            [
                "slice_value",
                "n",
                "delta_mean",
                "flip_rate",
                "wilcoxon_p_holm",
                "wilcoxon_reject_holm",
                "cohens_h",
            ]
        ].sort_values("delta_mean").to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )


if __name__ == "__main__":
    main()

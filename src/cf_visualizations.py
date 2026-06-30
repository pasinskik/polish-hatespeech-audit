"""Generuje 3 figury do paperu na podstawie cf_ptaszynski_*.parquet.

1. fig_heatmap_per_functionality.pdf — heatmapa Δp dla 4 par porównań × 27 functionality
2. fig_delta_per_target.pdf — bar plot per target_ident dla fem_vs_masc z 95% CI
3. fig_p_distribution.pdf — histogram P(harmful) per prefix_class

Wszystkie wektorowe (PDF), gotowe do \\includegraphics w LaTeX.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def load_data(model_key: str):
    preds_path = RESULTS_DIR / f"cf_{model_key}_predictions.parquet"
    stats_path = RESULTS_DIR / f"cf_{model_key}_statistical_tests.parquet"

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistical tests file: {stats_path}")

    preds = pd.read_parquet(preds_path)
    stats = pd.read_parquet(stats_path)
    return preds, stats


def fig_heatmap(stats: pd.DataFrame, model_key: str, model_label: str) -> Path:
    """Heatmapa: pary porównań × functionality, kolor = Δp."""
    pairs = ["fem_vs_orig", "masc_vs_orig", "neutral_vs_orig", "fem_vs_masc"]
    pair_labels = {
        "fem_vs_orig": "fem − original",
        "masc_vs_orig": "masc − original",
        "neutral_vs_orig": "neutral − original",
        "fem_vs_masc": "fem − masc",
    }

    sub = stats[(stats["slice_type"] == "functionality") & (stats["pair"].isin(pairs))]
    pivot = sub.pivot(index="slice_value", columns="pair", values="delta_mean")
    pivot = pivot[pairs]
    pivot = pivot.sort_values("fem_vs_orig")

    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    vmax = float(np.nanmax(np.abs(pivot.to_numpy())))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([pair_labels[p] for p in pairs], rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.02f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(val) > vmax * 0.55 else "black",
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"mean $\Delta p$ (probability shift)")
    ax.set_title(f"Δp per functionality ({model_label})")
    ax.set_xlabel("Comparison pair")
    ax.set_ylabel("HateCheck functionality")

    out = FIG_DIR / f"fig_{model_key}_heatmap_per_functionality.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_delta_per_target(stats: pd.DataFrame, model_key: str, model_label: str) -> Path:
    """Bar plot per target_ident dla pary fem_vs_masc z 95% CI."""
    sub = stats[(stats["slice_type"] == "target_ident") & (stats["pair"] == "fem_vs_masc")].copy()
    sub = sub[sub["n"] > 10].copy()
    sub = sub.replace({"__none__": "no target"})
    sub = sub.sort_values("delta_mean")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    y = np.arange(len(sub))
    means = sub["delta_mean"].to_numpy()
    ci_low = sub["delta_ci_low"].to_numpy()
    ci_high = sub["delta_ci_high"].to_numpy()
    err = np.vstack([means - ci_low, ci_high - means])

    ax.barh(
        y,
        means,
        xerr=err,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
        error_kw=dict(ecolor="black", capsize=2, elinewidth=0.8),
    )

    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["slice_value"])
    ax.set_xlabel(r"$\Delta p$ (feminine − masculine, mean with 95% CI)")
    ax.set_title(f"Effect of feminine vs masculine name by target group ({model_label})")
    ax.invert_yaxis()

    out = FIG_DIR / f"fig_{model_key}_delta_per_target.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_p_distribution(preds: pd.DataFrame, model_key: str, model_label: str) -> Path:
    """Histogram (overlapping density-like) P(harmful) per prefix_class."""
    classes = ["original", "neutral", "feminine", "masculine"]
    colors = {
        "original": "#444444",
        "neutral": "#888888",
        "feminine": "#d62728",
        "masculine": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 41)

    for cls in classes:
        s = preds.loc[preds["prefix_class"] == cls, "p_harmful"].to_numpy()
        ax.hist(
            s,
            bins=bins,
            alpha=0.45,
            label=f"{cls} (mean={s.mean():.2f})",
            color=colors[cls],
            edgecolor=colors[cls],
            linewidth=0.6,
            density=True,
        )

    ax.set_xlabel(r"$P(\mathrm{harmful})$")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of $P(\\mathrm{{harmful}})$ by prefix class ({model_label})")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 1)

    out = FIG_DIR / f"fig_{model_key}_p_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    classes = ["original", "neutral", "feminine", "masculine"]
    colors = {
        "original": "#444444",
        "neutral": "#888888",
        "feminine": "#d62728",
        "masculine": "#1f77b4",
    }

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.linspace(0, 1, 41)

    for cls in classes:
        s = preds.loc[preds["prefix_class"] == cls, "p_harmful"].to_numpy()
        ax.hist(
            s,
            bins=bins,
            alpha=0.45,
            label=f"{cls} (mean={s.mean():.2f})",
            color=colors[cls],
            edgecolor=colors[cls],
            linewidth=0.6,
            density=True,
        )

    ax.set_xlabel(r"$P(\mathrm{harmful})$")
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of $P(\\mathrm{{harmful}})$ by prefix class ({model_label})")
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(0, 1)

    out = FIG_DIR / f"fig_{model_key}_p_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()

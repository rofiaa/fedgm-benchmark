"""Figures for E1 (docs/privacy_e1.md), from results/privacy/e1_mia/{cells,roc}.csv.

    python experiments/privacy/plot_e1.py   # -> results/privacy/e1_mia/figs/*.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "results" / "privacy" / "e1_mia"
FIG = IN / "figs"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
CAT = {"gaussian": "#2a78d6", "lognormal": "#eb6834", "ist": "#1baf7a"}   # slots 1-3
MARK = {"gaussian": "o", "lognormal": "s", "ist": "^"}
POP_LABEL = {"gaussian": "Gaussian", "lognormal": "Log-normal (long tails)", "ist": "IST (real patients)"}
REL_LABEL = {"mean_var": "Means and variances (D = 2d)",
             "cov": "Means and covariance (D = d + d(d+1)/2)",
             "quad": "FedDSM fold sums, quadratic basis"}
SEQ = ["#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]               # blue 300-700

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 5,
})


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK2)


def _end_labels(ax, items, gap=0.06):
    """Right-end labels (x, y, text), nudged apart vertically (axes fraction) so none overlap."""
    to_ax = ax.transAxes.inverted()
    pts = sorted(((to_ax.transform(ax.transData.transform((x, y)))[1], x, y, t)
                  for x, y, t in items))
    placed = []
    for fy, *_ in pts:                       # upward pass: keep the gap
        placed.append(max(fy, placed[-1] + gap) if placed else fy)
    top = 0.97                               # downward pass: stay inside the axes
    for i in range(len(placed) - 1, -1, -1):
        placed[i] = min(placed[i], top)
        top = placed[i] - gap
    for fy, (_, x, y, t) in zip(placed, pts):
        ax.annotate(t, (x, y), xytext=(1.0, fy), textcoords=("axes fraction"),
                    xycoords="data", va="center", fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))


def by_n(cells, metric, pred, ylabel, fname, ylog=False, floor=None):
    """One panel per release, Gaussian population; a line per d, dashed = prediction."""
    g = cells[cells.population == "gaussian"]
    rels = [r for r in REL_LABEL if r in set(g.release)]
    fig, axes = plt.subplots(1, len(rels), figsize=(4.6 * len(rels), 3.6), sharey=True)
    for ax, rel in zip(np.atleast_1d(axes), rels):
        sub = g[g.release == rel]
        ds = sorted(sub.d.unique())
        labels = []
        for i, d in enumerate(ds):
            c = SEQ[i + (len(SEQ) - len(ds))]
            s = sub[sub.d == d].sort_values("n")
            ax.plot(s.n, s[metric], color=c, marker="o")
            ax.plot(s.n, s[pred], color=c, linestyle=(0, (3, 2)), linewidth=1.2)
            labels.append((s.n.iloc[-1], s[metric].iloc[-1], f"d = {d}"))
        if floor is not None:
            ax.axhline(floor, color=INK2, linewidth=1, linestyle=":")
        ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(REL_LABEL[rel], loc="left", fontsize=10)
        ax.set_xlabel("Patients in the released group, n")
        ax.set_xlim(8, 3000)
        fig.canvas.draw()
        _end_labels(ax, labels)
    np.atleast_1d(axes)[0].set_ylabel(ylabel)
    fig.text(0.01, 0.005, "Solid: attack on simulated data (Gaussian covariates). Dashed: "
             "Gaussian prediction (AUROC Φ(√(D/2n)); TPR 1 − Φ(z₁₋α − √(D/n))). "
             "Dotted: random guessing.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def collapse(cells):
    """All cells against D_eff / n; the prediction is one curve."""
    x = np.geomspace(cells.D_eff.min() / cells.n.max() / 1.5, 300, 200)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, metric, pred, title in [
            (axes[0], "auroc", lambda r: norm.cdf(np.sqrt(r / 2)), "AUROC"),
            (axes[1], "tpr_at_0.001", lambda r: 1 - norm.cdf(norm.ppf(0.999) - np.sqrt(r)),
             "TPR at 0.1% FPR")]:
        for p in CAT:
            s = cells[cells.population == p]
            ax.scatter(s.D_eff / s.n, s[metric], s=18, color=CAT[p], marker=MARK[p],
                       edgecolors=SURFACE, linewidths=0.6, label=POP_LABEL[p], zorder=3)
        ax.plot(x, pred(x), color=INK2, linewidth=1.5, label="Prediction (Gaussian)", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Released numbers per patient, D / n")
        ax.set_title(title, loc="left")
        if metric != "auroc":
            ax.set_yscale("log")
            ax.axhline(1e-3, color=INK2, linewidth=1, linestyle=":")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.text(0.01, 0.005, "Every cell of the grid: 3 populations × 3 releases × d × n. "
             "Dotted: random guessing.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "collapse_D_over_n.png", dpi=200)
    plt.close(fig)


def roc_panel(roc, cells):
    """Log-log ROC, covariance release, d = 10, one line per n, per population."""
    ns = [10, 35, 100, 650, 2000]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, p in zip(axes, CAT):
        for i, n in enumerate(ns):
            s = roc[(roc.population == p) & (roc.release == "cov") & (roc.d == 10) & (roc.n == n)]
            if s.empty:
                continue
            ax.plot(s.fpr, s.tpr.clip(lower=1e-4), color=SEQ[i], label=f"n = {n}")
        ax.plot([1e-4, 1], [1e-4, 1], color=INK2, linewidth=1, linestyle=":")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(1e-4, 1), ax.set_ylim(1e-4, 1.05)
        ax.set_xlabel("False-positive rate")
        ax.set_title(POP_LABEL[p], loc="left", fontsize=10)
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, 0.005, "Means and covariance of d = 10 covariates (D = 65). "
             "Dotted: random guessing.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "roc_cov_d10.png", dpi=200)
    plt.close(fig)


def mean_var_vs_cov(cells):
    """Bret's question: means and variances vs full covariance, d = 10, per population."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, p in zip(axes, CAT):
        for rel, ls in [("mean_var", (0, (3, 2))), ("cov", "-")]:
            s = cells[(cells.population == p) & (cells.release == rel) & (cells.d == 10)]
            s = s.sort_values("n")
            ax.plot(s.n, s["tpr_at_0.01"], color=CAT[p], linestyle=ls, marker=MARK[p])
            _end_label(ax, s.n.iloc[-1], s["tpr_at_0.01"].iloc[-1],
                       "mean + var" if rel == "mean_var" else "covariance", CAT[p])
        ax.axhline(0.01, color=INK2, linewidth=1, linestyle=":")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_xlim(8, 8000)
        ax.set_xlabel("Patients in the released group, n")
        ax.set_title(POP_LABEL[p], loc="left", fontsize=10)
    axes[0].set_ylabel("TPR at 1% FPR")
    fig.text(0.01, 0.005, "d = 10. Dashed: means and variances (D = 20). Solid: means and "
             "covariance (D = 65). Dotted: random guessing.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "mean_var_vs_cov_d10.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(IN / "cells.csv")
    roc = pd.read_csv(IN / "roc.csv")
    by_n(cells, "auroc", "auroc_pred", "Membership-inference AUROC", "auroc_vs_n.png",
         floor=0.5)
    by_n(cells, "tpr_at_0.01", "tpr_at_0.01_pred", "TPR at 1% FPR", "tpr1_vs_n.png",
         ylog=True, floor=0.01)
    collapse(cells)
    roc_panel(roc, cells)
    mean_var_vs_cov(cells)
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()

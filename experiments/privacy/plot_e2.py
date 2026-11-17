"""Figures for E2 (docs/privacy_e2.md), from results/privacy/e2_fedgm/cells.csv.

    python experiments/privacy/plot_e2.py   # -> results/privacy/e2_fedgm/figs/*.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.privacy.plot_e1 import GRID, INK2, SEQ, SURFACE, _end_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "results" / "privacy" / "e2_fedgm"
FIG = IN / "figs"
SIG_LABEL = {"treated_raw": "Draft's normaliser S_t (singular on IST)",
             "treated_shrunk": "Shrunk treated covariance (full rank)"}
NT_ORDER = ["10", "20", "31", "50", "all"]


def _nt_color(label):
    return SEQ[NT_ORDER.index(str(label))]


def _nt_text(label):
    return "all treated (~235)" if str(label) == "all" else f"n_t = {label}"


def by_evaluations(c):
    """Re-identification and exactly recovered values vs evaluations seen."""
    ct = c[c.attack == "cross_terms"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey="row")
    for col, sig in enumerate(SIG_LABEL):
        for row, (metric, ylabel) in enumerate([("reid_share", "Controls re-identified"),
                                                ("exact_value_share",
                                                 "Covariate values recovered exactly")]):
            ax = axes[row, col]
            labels = []
            for nt in NT_ORDER:
                s = ct[(ct.sigma == sig) & (ct.n_t_label.astype(str) == nt)]
                g = s.groupby("G")[metric].mean()
                ax.plot(g.index, g.values, color=_nt_color(nt), marker="o")
                labels.append((g.index[-1], g.values[-1], _nt_text(nt)))
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            if row == 0:
                ax.set_title(SIG_LABEL[sig], loc="left", fontsize=10)
                base = ct[ct.sigma == sig].reid_baseline.mean()
                ax.axhline(base, color=INK2, linewidth=1, linestyle=":")
            if row == 1:
                ax.set_xlabel("Search evaluations seen by the coordinator")
            if col == 0:
                ax.set_ylabel(ylabel)
            fig.canvas.draw()
            _end_labels(ax, labels, gap=0.07)
    fig.text(0.01, 0.005, "IST single-arm, 31 covariates, 10 datasets (mean). Attack: least "
             "squares on the cross terms B. Dotted: random re-identification.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.9, 1))
    fig.savefig(FIG / "reconstruction_vs_evaluations.png", dpi=200)
    plt.close(fig)


def moments(c):
    """Controls identified exactly from the matched-set sums alone."""
    mm = c[c.attack == "matched_moments"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, sig in enumerate(SIG_LABEL):
        for j, nt in enumerate(NT_ORDER):
            s = mm[(mm.sigma == sig) & (mm.n_t_label.astype(str) == nt)]
            x = j + (i - 0.5) * 0.28
            ax.scatter([x] * len(s), s.identified_share_of_matched, s=16,
                       color=SEQ[1] if i == 0 else SEQ[4], marker="o" if i == 0 else "s",
                       edgecolors=SURFACE, linewidths=0.6,
                       label=SIG_LABEL[sig] if j == 0 else None, zorder=3)
    ax.set_xticks(range(len(NT_ORDER)), [_nt_text(n) for n in NT_ORDER], fontsize=8.5)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Ever-matched controls\nidentified exactly")
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.01, 0.005, "From the matched controls' sums Σx alone (draft eq. 9).\n"
             "One dot per dataset.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "matched_moments_identified.png", dpi=200)
    plt.close(fig)


def dp_tradeoff(c):
    """The draft's DP (eq. 14): matches kept (utility) and re-identification (leakage)."""
    dp = c[(c.attack == "cross_terms_dp") & (c.sigma == "treated_shrunk")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (metric, G, ylabel) in zip(axes, [
            ("match_agreement", 1, "Matches unchanged by the noise"),
            ("reid_share", 100, "Controls re-identified (100 evaluations)")]):
        labels = []
        for nt in NT_ORDER:
            s = dp[(dp.n_t_label.astype(str) == nt) & (dp.G == G)]
            g = s.groupby("eps")[metric].mean()
            ax.plot(g.index, g.values, color=_nt_color(nt), marker="o")
            labels.append((g.index[-1], g.values[-1], _nt_text(nt)))
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("ε per entry of B, per evaluation")
        ax.set_ylabel(ylabel)
        if metric == "match_agreement":
            ax.text(0.5, 0.5, "≈ 0 for every cohort size\n(all five lines overlap)",
                    transform=ax.transAxes, ha="center", fontsize=9, color=INK2)
        else:
            fig.canvas.draw()
            _end_labels(ax, labels, gap=0.07)
    fig.text(0.01, 0.005, "Shrunk normaliser. Laplace noise with the draft's eq.-14 scale "
             "(ρ = the control pool's diameter). Per-patient ε by composition = "
             "evaluations × n_t × ε.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.9, 1))
    fig.savefig(FIG / "dp_tradeoff.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(IN / "cells.csv")
    by_evaluations(c)
    moments(c)
    dp_tradeoff(c)
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()

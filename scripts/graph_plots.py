#!/usr/bin/env python3
"""
Generate PDF figures for the CSE 151B/251B NanoGPT final report.

Usage:
    python scripts/report_figures.py --outdir figures

The script writes:
    figures/fig_training_trajectory.pdf
    figures/fig_data_mix_ablation.pdf
    figures/fig_final_data_mix.pdf
    figures/fig_endgame_comparison.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_training_trajectory(outdir: Path) -> None:
    stages = [
        "46.6M\n6k",
        "92.7M\n10k",
        "97.7M\n15k",
        "97.7M\n30k",
        "98.6M\n62k",
    ]
    ppl = np.array([76.4277, 36.5558, 29.5006, 25.8011, 23.6276])
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.plot(x, ppl, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Validation PPL")
    ax.set_xlabel("Training stage")
    ax.set_title("Validation perplexity across training stages")
    ax.grid(True, linewidth=0.4, alpha=0.5)

    for xi, yi in zip(x, ppl):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_training_trajectory.pdf")
    plt.close(fig)


def save_data_mix_ablation(outdir: Path) -> None:
    labels = ["D0", "D1", "D2", "D3", "D5"]
    ppl = np.array([227.0333, 221.3258, 238.9537, 231.4606, 232.9486])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    ax.bar(x, ppl)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Validation PPL")
    ax.set_xlabel("Data mixture")
    ax.set_title("Small-model data mixture sweep")
    ax.set_ylim(215, 245)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    best_idx = int(np.argmin(ppl))
    for xi, yi in zip(x, ppl):
        label = f"{yi:.1f}"
        weight = "bold" if xi == best_idx else "normal"
        ax.annotate(label, (xi, yi), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, fontweight=weight)

    fig.tight_layout()
    fig.savefig(outdir / "fig_data_mix_ablation.pdf")
    plt.close(fig)


def save_final_data_mix(outdir: Path) -> None:
    sources = [
        "FineWeb-Edu",
        "OpenWebText",
        "Wikimedia",
        "arXiv",
        "PG-19",
        "StackExchange",
    ]
    weights = np.array([58, 16, 12, 8, 4, 2])
    y = np.arange(len(sources))

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.barh(y, weights)
    ax.set_yticks(y)
    ax.set_yticklabels(sources)
    ax.invert_yaxis()
    ax.set_xlabel("Mixture weight (%)")
    ax.set_title("Final recovery data mixture")
    ax.grid(axis="x", linewidth=0.4, alpha=0.5)

    for yi, wi in zip(y, weights):
        ax.annotate(f"{wi}%", (wi, yi), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_final_data_mix.pdf")
    plt.close(fig)


def save_endgame_comparison(outdir: Path) -> None:
    labels = ["Stable\ncheckpoint", "Cold broad\nshift", "Warm-restart\nrecovery"]
    ppl = np.array([25.8011, 27.1992, 23.6276])
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.bar(x, ppl)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Validation PPL")
    ax.set_title("Endgame continuation outcomes")
    ax.set_ylim(22.5, 28.0)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for xi, yi in zip(x, ppl):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_endgame_comparison.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
        help="Directory where PDF figures will be written.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    save_training_trajectory(args.outdir)
    save_data_mix_ablation(args.outdir)
    save_final_data_mix(args.outdir)
    save_endgame_comparison(args.outdir)

    print(f"Wrote report figures to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

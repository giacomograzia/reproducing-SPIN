"""
Experiment 1 figure: paired-dot/slopegraph showing the 25 individual
run-level observations (not just aggregate bars) for relevant vs unrelated
N=1 localization, fairness and privacy side by side. Each pair (relevant
run i, unrelated run i) is connected by a thin line so the paired structure
is visible.

Style matches reproduction/figures_and_tables/FT-data-scale/plot_results.py:
Agg backend, PNG (300dpi, bbox_inches='tight') + PDF, fairness/privacy color
convention (#E91E63 / #009688) consistent with that script's
plot_spin_extra_results.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N1_ROOT = Path(__file__).resolve().parents[1]

COLOR_RELEVANT = "#FF8A65"   # matches existing "SPIN" orange
COLOR_UNRELATED = "#9E9E9E"  # neutral gray for the control condition
COLOR_LINE = "#BDBDBD"


def plot_paired(ax, relevant_vals, unrelated_vals, title, ylabel):
    n = len(relevant_vals)
    x_relevant, x_unrelated = 0, 1

    for r, u in zip(relevant_vals, unrelated_vals):
        ax.plot([x_relevant, x_unrelated], [r, u], color=COLOR_LINE, linewidth=0.8, alpha=0.6, zorder=1)

    ax.scatter([x_relevant] * n, relevant_vals, color=COLOR_RELEVANT, s=45, zorder=2, label="Relevant N=1", edgecolor="white", linewidth=0.5)
    ax.scatter([x_unrelated] * n, unrelated_vals, color=COLOR_UNRELATED, s=45, zorder=2, label="Unrelated N=1", edgecolor="white", linewidth=0.5)

    mean_r = sum(relevant_vals) / n
    mean_u = sum(unrelated_vals) / n
    ax.plot([x_relevant - 0.08, x_relevant + 0.08], [mean_r, mean_r], color=COLOR_RELEVANT, linewidth=3, zorder=3)
    ax.plot([x_unrelated - 0.08, x_unrelated + 0.08], [mean_u, mean_u], color=COLOR_UNRELATED, linewidth=3, zorder=3)

    ax.set_xticks([x_relevant, x_unrelated])
    ax.set_xticklabels(["Relevant\nN=1", "Unrelated\nN=1"], fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--paired_json", type=str, default=None, help="Default: analysis/<model>/experiment1_paired.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/<model>")
    args = parser.parse_args()

    if args.paired_json is None:
        args.paired_json = str(N1_ROOT / "analysis" / args.model / "experiment1_paired.json")
    if args.out_dir is None:
        args.out_dir = str(N1_ROOT / "figures" / args.model)

    with open(args.paired_json) as f:
        rows = json.load(f)
    if not rows:
        raise SystemExit(f"No paired runs found in {args.paired_json}")

    fairness_relevant = [r["fairness_relevant"] for r in rows]
    fairness_unrelated = [r["fairness_unrelated"] for r in rows]
    privacy_relevant = [r["privacy_relevant"] for r in rows]
    privacy_unrelated = [r["privacy_unrelated"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_paired(axes[0], fairness_relevant, fairness_unrelated, "Fairness Awareness: Relevant vs Unrelated N=1", "Awareness Rate")
    plot_paired(axes[1], privacy_relevant, privacy_unrelated, "Privacy Awareness: Relevant vs Unrelated N=1", "Awareness Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Experiment 1: Semantic-Relevance Control -- {args.model} (n={len(rows)} paired runs)", fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "experiment1_paired_relevant_vs_unrelated.png"
    out_pdf = out_dir / "experiment1_paired_relevant_vs_unrelated.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()

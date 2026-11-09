"""
Experiment 2 figures: scatter plots of final-coupled-set Jaccard similarity
(x-axis, primary stability metric) against fairness/privacy delta vs the
origin baseline (y-axis), one point per N=1 run. Deliberately NO regression
/trend line -- only Spearman correlation (rank-based) is being tested, not
a linear relationship, per PLAN.md's explicit instruction not to imply
linear regression.

Style matches reproduction/figures_and_tables/FT-data-scale/plot_results.py:
Agg backend, PNG (300dpi, bbox_inches='tight') + PDF.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N1_ROOT = Path(__file__).resolve().parents[1]

COLOR_FAIRNESS = "#E91E63"
COLOR_PRIVACY = "#009688"


def scatter_plot(ax, x, y, color, title, ylabel, rho=None, p=None):
    ax.scatter(x, y, color=color, s=55, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=1)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_xlabel("Final Coupled-Set Jaccard Similarity vs Reference", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.8)
    if rho is not None:
        ax.text(
            0.03, 0.97, f"Spearman rho = {rho:.3f}\npermutation p = {p:.3f}",
            transform=ax.transAxes, fontsize=11, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="lightgray"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--jaccard_json", type=str, default=None, help="Default: analysis/<model>/jaccard_stability_results.json")
    parser.add_argument("--relevant_summary", type=str, default=None, help="Default: analysis/<model>/results_relevant_n1_summary.json")
    parser.add_argument("--stats_json", type=str, default=None, help="Default: analysis/<model>/stability_vs_performance_stats.json")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: figures/<model>")
    args = parser.parse_args()

    if args.jaccard_json is None:
        args.jaccard_json = str(N1_ROOT / "analysis" / args.model / "jaccard_stability_results.json")
    if args.relevant_summary is None:
        args.relevant_summary = str(N1_ROOT / "analysis" / args.model / "results_relevant_n1_summary.json")
    if args.stats_json is None:
        args.stats_json = str(N1_ROOT / "analysis" / args.model / "stability_vs_performance_stats.json")
    if args.out_dir is None:
        args.out_dir = str(N1_ROOT / "figures" / args.model)

    with open(args.jaccard_json) as f:
        jaccard_rows = {r["run_id"]: r for r in json.load(f)}
    with open(args.relevant_summary) as f:
        relevant_rows = {r["run_id"]: r for r in json.load(f)}

    stats = None
    if Path(args.stats_json).exists():
        with open(args.stats_json) as f:
            stats = json.load(f)

    common_ids = sorted(set(jaccard_rows) & set(relevant_rows))
    if not common_ids:
        raise SystemExit("No runs with both Jaccard and eval results found.")

    jaccard = [jaccard_rows[i]["final_coupled_jaccard"] for i in common_ids]
    fairness_delta = [relevant_rows[i]["fairness_delta_vs_origin"] for i in common_ids]
    privacy_delta = [relevant_rows[i]["privacy_delta_vs_origin"] for i in common_ids]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    f_rho = stats["fairness_delta_vs_jaccard"]["rho"] if stats else None
    f_p = stats["fairness_delta_vs_jaccard"]["permutation_p_value"] if stats else None
    p_rho = stats["privacy_delta_vs_jaccard"]["rho"] if stats else None
    p_p = stats["privacy_delta_vs_jaccard"]["permutation_p_value"] if stats else None

    scatter_plot(axes[0], jaccard, fairness_delta, COLOR_FAIRNESS,
                 "Localization Stability vs Fairness Gain", "Fairness Delta vs Origin", f_rho, f_p)
    scatter_plot(axes[1], jaccard, privacy_delta, COLOR_PRIVACY,
                 "Localization Stability vs Privacy Gain", "Privacy Delta vs Origin", p_rho, p_p)

    fig.suptitle(f"Experiment 2: N=1 Localization Stability vs Downstream Performance -- {args.model} (n={len(common_ids)} runs)",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    out_png = out_dir / "experiment2_jaccard_vs_fairness_privacy_delta.png"
    out_pdf = out_dir / "experiment2_jaccard_vs_fairness_privacy_delta.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved plots to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()

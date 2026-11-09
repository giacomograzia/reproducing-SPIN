"""
Cross-model grouped figures: combines each model's individual Experiment 1 /
Experiment 2 figure into a single 1xN row (one panel per model), for
side-by-side comparison. Reuses the exact per-panel plotting helpers from
plot_experiment1.py (plot_paired) and plot_experiment2_stability.py
(scatter_plot) so visual style is identical to the per-model figures --
no plotting logic is duplicated here, only the panel layout/composition.

Produces four figures (two per experiment, one per metric -- fairness and
privacy are kept as separate 1xN figures rather than combined into one,
since combining both metrics into a single panel would either double the
points per panel or require a 2xN grid, and a plain 1xN-per-metric figure
is the most direct "all three models' Experiment N visualization side by
side" layout):
  - grouped_figures/experiment1_fairness_grouped.{png,pdf}
  - grouped_figures/experiment1_privacy_grouped.{png,pdf}
  - grouped_figures/experiment2_fairness_grouped.{png,pdf}
  - grouped_figures/experiment2_privacy_grouped.{png,pdf}

Requires each included model's analysis/<model>/ outputs to already exist
(i.e. that model's run_analysis.job must have completed first). Models
with missing data are skipped with a warning, not silently omitted without
notice -- run with --models to restrict to a specific subset (e.g. while
other models' sweeps are still in progress).

CPU-only, reads small JSON files -- submitted via sbatch per the project's
compute policy (see jobs/run_grouped_analysis.job).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N1_ROOT = Path(__file__).resolve().parents[1]
if str(N1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(N1_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model_registry import load_models  # noqa: E402
from plot_experiment1 import plot_paired  # noqa: E402
from plot_experiment2_stability import COLOR_FAIRNESS, COLOR_PRIVACY, scatter_plot  # noqa: E402

DEFAULT_OUT_DIR = N1_ROOT / "figures" / "grouped_figures"


def available_models(requested_models):
    """Filters requested_models down to those with a complete experiment1_paired.json
    (used as the readiness signal for both experiments, since it's produced early
    in build_summary_tables.py and both experiments' analysis scripts depend on
    the same upstream run_result.json files)."""
    ready, missing = [], []
    for model in requested_models:
        if (N1_ROOT / "analysis" / model / "experiment1_paired.json").exists():
            ready.append(model)
        else:
            missing.append(model)
    if missing:
        print(f"[plot_grouped_comparison] WARNING: skipping models with no analysis output yet: {missing}")
    return ready


def load_experiment1_paired(model):
    with open(N1_ROOT / "analysis" / model / "experiment1_paired.json") as f:
        return json.load(f)


def plot_experiment1_grouped(models, metric, out_dir):
    """metric: 'fairness' or 'privacy'."""
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5.5), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        rows = load_experiment1_paired(model)
        relevant_vals = [r[f"{metric}_relevant"] for r in rows]
        unrelated_vals = [r[f"{metric}_unrelated"] for r in rows]
        plot_paired(ax, relevant_vals, unrelated_vals, f"{model}\n(n={len(rows)} paired runs)", "Awareness Rate")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"Experiment 1: {metric.capitalize()} Awareness -- Relevant vs Unrelated N=1, Across Models", fontsize=15, y=1.04)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"experiment1_{metric}_grouped.png"
    out_pdf = out_dir / f"experiment1_{metric}_grouped.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")


def load_experiment2_data(model):
    with open(N1_ROOT / "analysis" / model / "jaccard_stability_results.json") as f:
        jaccard_rows = {r["run_id"]: r for r in json.load(f)}
    with open(N1_ROOT / "analysis" / model / "results_relevant_n1_summary.json") as f:
        relevant_rows = {r["run_id"]: r for r in json.load(f)}

    stats = None
    stats_path = N1_ROOT / "analysis" / model / "stability_vs_performance_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    common_ids = sorted(set(jaccard_rows) & set(relevant_rows))
    jaccard = [jaccard_rows[i]["final_coupled_jaccard"] for i in common_ids]
    fairness_delta = [relevant_rows[i]["fairness_delta_vs_origin"] for i in common_ids]
    privacy_delta = [relevant_rows[i]["privacy_delta_vs_origin"] for i in common_ids]
    return jaccard, fairness_delta, privacy_delta, stats, len(common_ids)


def plot_experiment2_grouped(models, metric, out_dir):
    """metric: 'fairness' or 'privacy'."""
    color = COLOR_FAIRNESS if metric == "fairness" else COLOR_PRIVACY
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5), squeeze=False)
    axes = axes[0]

    for ax, model in zip(axes, models):
        jaccard, fairness_delta, privacy_delta, stats, n = load_experiment2_data(model)
        y = fairness_delta if metric == "fairness" else privacy_delta
        rho = p = None
        if stats:
            entry = stats.get(f"{metric}_delta_vs_jaccard")
            if entry:
                rho, p = entry["rho"], entry["permutation_p_value"]
        scatter_plot(ax, jaccard, y, color, f"{model}\n(n={n} runs)", f"{metric.capitalize()} Delta vs Origin", rho, p)

    fig.suptitle(f"Experiment 2: Localization Stability vs {metric.capitalize()} Gain, Across Models", fontsize=15, y=1.04)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"experiment2_{metric}_grouped.png"
    out_pdf = out_dir / f"experiment2_{metric}_grouped.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png} and {out_pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", default=None,
                         help="Model labels to include, in display order. Default: all labels in configs/models.yaml.")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    requested_models = args.models if args.models else list(load_models().keys())
    models = available_models(requested_models)
    if len(models) < 1:
        raise SystemExit("No models with completed analysis output found -- run run_analysis.job for at least one model first.")

    out_dir = Path(args.out_dir)

    plot_experiment1_grouped(models, "fairness", out_dir)
    plot_experiment1_grouped(models, "privacy", out_dir)
    plot_experiment2_grouped(models, "fairness", out_dir)
    plot_experiment2_grouped(models, "privacy", out_dir)


if __name__ == "__main__":
    main()

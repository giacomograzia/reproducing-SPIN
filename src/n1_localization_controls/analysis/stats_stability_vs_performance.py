"""
Experiment 2, "Localization Stability vs Downstream Performance": tests
whether stronger agreement with the full-data reference localization
(final-coupled-set Jaccard similarity) is associated with better downstream
fairness/privacy performance, across the same 25 relevant N=1 runs.

Reads:
  - analysis/<model>/jaccard_stability_results.json (compute_jaccard_stability.py)
  - analysis/<model>/results_relevant_n1_summary.json (build_summary_tables.py)
  - configs/models.yaml (origin baseline for --model, via model_registry.py)

Primary (Holm-corrected) pair:
  - Spearman(final_coupled_jaccard, fairness_delta_vs_origin)
  - Spearman(final_coupled_jaccard, privacy_delta_vs_origin)
Secondary/descriptive only (not in the Holm family):
  - Spearman(final_coupled_jaccard, joint_delta) where
    joint_delta = (fairness_delta + privacy_delta) / 2

Every Spearman result is reported as (rho, permutation p-value, bootstrap
CI) per stats_utils.py -- never scipy.stats.spearmanr's asymptotic p-value.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import holm_correction, spearman_bootstrap_ci, spearman_permutation_pvalue, spearman_rho  # noqa: E402

N1_ROOT = Path(__file__).resolve().parents[1]
if str(N1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(N1_ROOT / "scripts"))

from model_registry import get_model  # noqa: E402


def spearman_triple(x, y, n_resamples, seed):
    rho = spearman_rho(x, y)
    perm = spearman_permutation_pvalue(x, y, n_resamples=n_resamples, seed=seed)
    ci = spearman_bootstrap_ci(x, y, n_resamples=n_resamples, seed=seed)
    return {
        "rho": rho,
        "permutation_p_value": perm["p_value"],
        "bootstrap_ci_low": ci["ci_low"],
        "bootstrap_ci_high": ci["ci_high"],
        "n": len(x),
        "n_resamples": n_resamples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--jaccard_json", type=str, default=None, help="Default: analysis/<model>/jaccard_stability_results.json")
    parser.add_argument("--relevant_summary", type=str, default=None, help="Default: analysis/<model>/results_relevant_n1_summary.json")
    parser.add_argument("--config", type=str, default=str(N1_ROOT / "configs" / "n1_experiment_config.yaml"))
    parser.add_argument("--out_csv", type=str, default=None, help="Default: analysis/<model>/stability_vs_performance_stats.csv")
    parser.add_argument("--out_json", type=str, default=None, help="Default: analysis/<model>/stability_vs_performance_stats.json")
    args = parser.parse_args()

    if args.jaccard_json is None:
        args.jaccard_json = str(N1_ROOT / "analysis" / args.model / "jaccard_stability_results.json")
    if args.relevant_summary is None:
        args.relevant_summary = str(N1_ROOT / "analysis" / args.model / "results_relevant_n1_summary.json")
    if args.out_csv is None:
        args.out_csv = str(N1_ROOT / "analysis" / args.model / "stability_vs_performance_stats.csv")
    if args.out_json is None:
        args.out_json = str(N1_ROOT / "analysis" / args.model / "stability_vs_performance_stats.json")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    stats_cfg = cfg["stats"]
    get_model(args.model)  # validates --model against configs/models.yaml (fails fast on a typo'd label);
    # origin baseline itself isn't needed here -- fairness/privacy_delta_vs_origin
    # were already computed against it by run_n1_run.py and are read pre-computed below.

    with open(args.jaccard_json) as f:
        jaccard_rows = {r["run_id"]: r for r in json.load(f)}
    with open(args.relevant_summary) as f:
        relevant_rows = {r["run_id"]: r for r in json.load(f)}

    common_ids = sorted(set(jaccard_rows) & set(relevant_rows))
    if len(common_ids) < 3:
        raise SystemExit(f"Need at least 3 runs with both Jaccard and eval results, found {len(common_ids)}")

    jaccard = np.array([jaccard_rows[i]["final_coupled_jaccard"] for i in common_ids])
    fairness_delta = np.array([relevant_rows[i]["fairness_delta_vs_origin"] for i in common_ids])
    privacy_delta = np.array([relevant_rows[i]["privacy_delta_vs_origin"] for i in common_ids])
    joint_delta = (fairness_delta + privacy_delta) / 2.0

    n_res, seed = stats_cfg["n_permutations"], stats_cfg["seed"]
    fairness_result = spearman_triple(jaccard, fairness_delta, n_res, seed)
    privacy_result = spearman_triple(jaccard, privacy_delta, n_res, seed)
    joint_result = spearman_triple(jaccard, joint_delta, n_res, seed)  # secondary/descriptive

    raw_pvals = [fairness_result["permutation_p_value"], privacy_result["permutation_p_value"]]
    adjusted = holm_correction(raw_pvals, alpha=stats_cfg["holm_alpha"])
    fairness_result["holm_adjusted_p_value"] = float(adjusted[0])
    privacy_result["holm_adjusted_p_value"] = float(adjusted[1])
    joint_result["holm_adjusted_p_value"] = None  # secondary/descriptive -- not in the Holm family

    out = {
        "primary_metric": "final_coupled_jaccard",
        "fairness_delta_vs_jaccard": fairness_result,
        "privacy_delta_vs_jaccard": privacy_result,
        "joint_delta_vs_jaccard_descriptive_only": joint_result,
        "n_runs": len(common_ids),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "rho", "permutation_p", "holm_adjusted_p", "ci_low", "ci_high", "n"])
        for name, d in [
            ("jaccard_vs_fairness_delta", fairness_result),
            ("jaccard_vs_privacy_delta", privacy_result),
            ("jaccard_vs_joint_delta (descriptive)", joint_result),
        ]:
            writer.writerow([name, d["rho"], d["permutation_p_value"], d["holm_adjusted_p_value"],
                              d["bootstrap_ci_low"], d["bootstrap_ci_high"], d["n"]])

    print(f"\n{'Comparison':<38} {'rho':>8} {'perm p':>10} {'Holm p':>10} {'95% CI':>22}")
    print("-" * 92)
    for name, d in [
        ("final_coupled_jaccard vs fairness_delta", fairness_result),
        ("final_coupled_jaccard vs privacy_delta", privacy_result),
        ("final_coupled_jaccard vs joint_delta*", joint_result),
    ]:
        holm_str = f"{d['holm_adjusted_p_value']:.4f}" if d["holm_adjusted_p_value"] is not None else "n/a"
        print(f"{name:<38} {d['rho']:>8.4f} {d['permutation_p_value']:>10.4f} {holm_str:>10} "
              f"[{d['bootstrap_ci_low']:+.4f}, {d['bootstrap_ci_high']:+.4f}]")
    print("* joint_delta is secondary/descriptive only, excluded from the Holm family.")
    print(f"\nn = {len(common_ids)} runs; permutation/bootstrap resamples = {n_res}")
    print(f"\nSaved: {args.out_csv}\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()

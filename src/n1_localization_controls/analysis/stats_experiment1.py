"""
Experiment 1 statistical analysis: relevant vs unrelated N=1, paired across
the 25 runs (relevant run i <-> unrelated run i, same seed=run_id).

Reads analysis/experiment1_paired.csv (written by build_summary_tables.py).
For fairness and privacy separately, computes:
  - relevant mean/std, unrelated mean/std (ddof=1)
  - mean paired difference, median + IQR
  - two-sided paired permutation test p-value (>=10000 resamples)
  - 95% paired bootstrap CI for the mean paired difference (>=10000 resamples)
Then applies Holm correction across the two primary p-values (fairness,
privacy). Prints a console summary and saves CSV/JSON.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import holm_correction, paired_bootstrap_ci, paired_permutation_test  # noqa: E402

N1_ROOT = Path(__file__).resolve().parents[1]


def load_paired(path):
    with open(path) as f:
        return json.load(f)


def analyze_metric(rows, relevant_key, unrelated_key, n_permutations, n_bootstrap, seed):
    x = np.array([r[relevant_key] for r in rows], dtype=float)
    y = np.array([r[unrelated_key] for r in rows], dtype=float)
    diffs = x - y

    perm = paired_permutation_test(x, y, n_resamples=n_permutations, seed=seed)
    ci = paired_bootstrap_ci(x, y, n_resamples=n_bootstrap, seed=seed)

    q1, q3 = np.percentile(diffs, [25, 75])

    return {
        "n": len(rows),
        "relevant_mean": float(np.mean(x)),
        "relevant_std": float(np.std(x, ddof=1)),
        "unrelated_mean": float(np.mean(y)),
        "unrelated_std": float(np.std(y, ddof=1)),
        "mean_paired_diff": float(np.mean(diffs)),
        "median_paired_diff": float(np.median(diffs)),
        "iqr_low": float(q1),
        "iqr_high": float(q3),
        "permutation_p_value": perm["p_value"],
        "permutation_n_resamples": perm["n_resamples"],
        "bootstrap_ci_low": ci["ci_low"],
        "bootstrap_ci_high": ci["ci_high"],
        "bootstrap_n_resamples": ci["n_resamples"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--paired_json", type=str, default=None, help="Default: analysis/<model>/experiment1_paired.json")
    parser.add_argument("--config", type=str, default=str(N1_ROOT / "configs" / "n1_experiment_config.yaml"))
    parser.add_argument("--out_csv", type=str, default=None, help="Default: analysis/<model>/experiment1_stats.csv")
    parser.add_argument("--out_json", type=str, default=None, help="Default: analysis/<model>/experiment1_stats.json")
    args = parser.parse_args()

    if args.paired_json is None:
        args.paired_json = str(N1_ROOT / "analysis" / args.model / "experiment1_paired.json")
    if args.out_csv is None:
        args.out_csv = str(N1_ROOT / "analysis" / args.model / "experiment1_stats.csv")
    if args.out_json is None:
        args.out_json = str(N1_ROOT / "analysis" / args.model / "experiment1_stats.json")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    stats_cfg = cfg["stats"]

    rows = load_paired(args.paired_json)
    if len(rows) < 2:
        raise SystemExit(f"Need at least 2 paired runs, found {len(rows)} in {args.paired_json}")

    fairness = analyze_metric(
        rows, "fairness_relevant", "fairness_unrelated",
        stats_cfg["n_permutations"], stats_cfg["n_bootstrap"], stats_cfg["seed"],
    )
    privacy = analyze_metric(
        rows, "privacy_relevant", "privacy_unrelated",
        stats_cfg["n_permutations"], stats_cfg["n_bootstrap"], stats_cfg["seed"],
    )

    raw_pvals = [fairness["permutation_p_value"], privacy["permutation_p_value"]]
    adjusted = holm_correction(raw_pvals, alpha=stats_cfg["holm_alpha"])
    fairness["holm_adjusted_p_value"] = float(adjusted[0])
    privacy["holm_adjusted_p_value"] = float(adjusted[1])

    out = {"fairness": fairness, "privacy": privacy, "n_paired_runs": len(rows)}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + list(fairness.keys()))
        writer.writerow(["fairness"] + list(fairness.values()))
        writer.writerow(["privacy"] + list(privacy.values()))

    print(f"\n{'Metric':<10} {'Relevant':>18} {'Unrelated':>18} {'MeanDiff':>10} {'Perm p':>10} {'Holm p':>10} {'95% CI':>22}")
    print("-" * 100)
    for name, d in [("fairness", fairness), ("privacy", privacy)]:
        print(f"{name:<10} "
              f"{d['relevant_mean']:.4f}+/-{d['relevant_std']:.4f}   "
              f"{d['unrelated_mean']:.4f}+/-{d['unrelated_std']:.4f}   "
              f"{d['mean_paired_diff']:>+10.4f} {d['permutation_p_value']:>10.4f} {d['holm_adjusted_p_value']:>10.4f}   "
              f"[{d['bootstrap_ci_low']:+.4f}, {d['bootstrap_ci_high']:+.4f}]")
    print(f"\nn paired runs = {len(rows)}; permutation resamples = {stats_cfg['n_permutations']}; "
          f"bootstrap resamples = {stats_cfg['n_bootstrap']}; Holm alpha = {stats_cfg['holm_alpha']}")
    print(f"\nSaved: {args.out_csv}\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()

"""
Paired-bootstrap + paired-permutation comparison of each fine-tuning method
against the origin (untouched) model, on both safety (MD-Judge) and
helpfulness (Prometheus-2) metrics, for each model (Qwen/Mistral/Vicuna) and
each held-out test set (safety_test.csv, helpfulness_test.csv).

Reuses the paired_permutation_test / paired_bootstrap_ci / holm_correction
primitives already validated in
n1_localization_controls/analysis/stats_utils.py (paired permutation test
for p-values, percentile bootstrap for CIs) rather than reimplementing
paired bootstrap -- see that module's docstring for why
permutation_type='samples' is the correct null for a paired-difference test.

Pairing unit: each (seed, row-position) pair is one paired observation --
origin's score on that prompt vs. the fine-tuned checkpoint's score on that
same prompt (both runs generate from the identical, deterministically
ordered safety_test.csv / helpfulness_test.csv, temperature=0). All 3
training seeds are pooled into one test per (model, method, dataset,
metric), so the comparison also reflects training-seed variance in the
fine-tuned condition, not just per-prompt variance -- a single lucky/unlucky
seed can't drive the result on its own.

Usage:
    python stats_origin_vs_finetuned.py [--out out.csv] [--n_resamples 10000]

Requires the corresponding eval jobs (origin_safety_helpful_multiseed_*.job,
fft_safety_helpfulness_multiseed_*.job, lora_dora_safety_helpfulness_multiseed_*.job,
spin_safety_helpful_multiseed_*.job) to have actually completed and written
paired_results.csv -- this script does not run any model, only reads their
outputs. Any (model, method, dataset) combination whose files are missing is
skipped with a printed note, not an error, so partial result sets can be
analyzed as jobs finish.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "n1_localization_controls" / "analysis"))
from stats_utils import paired_permutation_test, paired_bootstrap_ci, holm_correction  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
EVAL_OUTPUTS = REPO_ROOT / "src" / "jobs" / "safety_helpfulness_paired" / "outputs"

MODELS = ["qwen", "mistral", "vicuna"]
MODEL_KEY = {"qwen": "qwen2_7b", "mistral": "mistral_7b", "vicuna": "vicuna_7b"}
DATASETS = ["safety_test", "helpfulness_test"]
SEEDS = [0, 1, 2]
METHODS = ["fft", "lora", "dora", "spin"]

# Origin generation is deterministic (temperature=0, greedy decoding, no
# training), so all 3 seeds of origin_safety_helpful_multiseed_*.job produce
# byte-identical output -- only seed 0 is actually run (seeds 1/2 were
# cancelled as redundant). Every fine-tuned method's seed is paired against
# this single origin result, not against a seed-matched origin file.
ORIGIN_SEED = 0


def origin_paired_results_path(eval_outputs, model, dataset):
    return eval_outputs / f"origin_{model}_multiseed" / f"seed_{ORIGIN_SEED}" / dataset / "paired_results.csv"


def method_paired_results_path(eval_outputs, model, method, dataset, seed):
    if method == "fft":
        return (eval_outputs / f"fft_{model}_safety_helpfulness_multiseed"
                 / f"seed_{seed}" / dataset / "paired_results.csv")
    if method == "spin":
        return eval_outputs / f"spin_{model}_multiseed" / f"seed_{seed}" / dataset / "paired_results.csv"
    if method in ("lora", "dora"):
        tag = f"{MODEL_KEY[model]}_{method}_safety_helpfulness_seed{seed}_merged"
        return (eval_outputs / f"lora_dora_{model}_safety_helpfulness_multiseed"
                 / f"seed_{seed}" / tag / dataset / "paired_results.csv")
    raise ValueError(f"unknown method: {method}")


def load_metric_series(csv_path, metric):
    """Metric value per row, invalid entries (-1: unparseable judge output /
    'no_following') mapped to NaN so they drop out of the paired comparison.
    """
    df = pd.read_csv(csv_path)
    if metric == "safety":
        # safety_label: 0=safe, 1=unsafe, -1=judge output didn't follow the
        # expected format. is_safe is the higher-is-better version actually
        # compared (1=safe/refused, 0=unsafe/complied).
        vals = df["safety_label"].astype(float)
        vals = vals.where(vals.isin([0.0, 1.0]))
        return 1.0 - vals
    if metric == "helpfulness":
        vals = df["helpfulness_score"].astype(float)
        return vals.where(vals != -1)
    raise ValueError(f"unknown metric: {metric}")


def pooled_paired_arrays(eval_outputs, model, method, dataset, metric):
    """Pool all 3 of the method's seeds' per-row paired (origin, method)
    values for one (model, method, dataset, metric) combination into two
    aligned 1-D arrays, dropping any row where either side is invalid (-1)
    or missing, and any seed whose method file doesn't exist yet.

    Origin is NOT re-read per seed: it has only one real result (seed 0 --
    see ORIGIN_SEED), reused as the baseline for every one of the method's
    seeds. So this is "1 origin run vs. each of the method's 3 independent
    training seeds," not seed-matched pairs on both sides.
    """
    origin_path = origin_paired_results_path(eval_outputs, model, dataset)
    if not origin_path.exists():
        return None, None, []
    x_origin = load_metric_series(origin_path, metric)

    x_all, y_all, seeds_used = [], [], []
    for seed in SEEDS:
        method_path = method_paired_results_path(eval_outputs, model, method, dataset, seed)
        if not method_path.exists():
            continue

        y = load_metric_series(method_path, metric)
        n = min(len(x_origin), len(y))
        if len(x_origin) != len(y):
            print(f"[WARN] {model}/{method}/{dataset}/seed{seed}: origin has {len(x_origin)} rows, "
                  f"{method} has {len(y)} rows -- truncating to first {n}. Both should have run "
                  f"the same source CSV with the same --max_samples; a mismatch here means that "
                  f"assumption broke somewhere -- verify before trusting this comparison.")
        x, y = x_origin.iloc[:n].to_numpy(), y.iloc[:n].to_numpy()

        valid = ~(np.isnan(x) | np.isnan(y))
        x_all.append(x[valid])
        y_all.append(y[valid])
        seeds_used.append(seed)

    if not x_all:
        return None, None, []
    return np.concatenate(x_all), np.concatenate(y_all), seeds_used


def compare_one(eval_outputs, model, method, dataset, metric, n_resamples=10000, seed=0):
    x, y, seeds_used = pooled_paired_arrays(eval_outputs, model, method, dataset, metric)
    if x is None or len(x) < 2:
        return None

    # method - origin: positive = the fine-tuned method scores higher.
    perm = paired_permutation_test(y, x, n_resamples=n_resamples, seed=seed)
    ci = paired_bootstrap_ci(y, x, n_resamples=n_resamples, seed=seed)
    diffs = y - x

    return {
        "model": model,
        "method": method,
        "dataset": dataset,
        "metric": metric,
        "n": len(x),
        "seeds_used": ",".join(str(s) for s in seeds_used),
        "origin_mean": float(np.mean(x)),
        "origin_std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        "method_mean": float(np.mean(y)),
        "method_std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "mean_diff": float(np.mean(diffs)),
        "diff_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan"),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "p_value": perm["p_value"],
        "n_resamples": n_resamples,
    }


def run_all(eval_outputs, n_resamples=10000, seed=0):
    rows, skipped = [], []
    for model in MODELS:
        for method in METHODS:
            for dataset in DATASETS:
                for metric in ("safety", "helpfulness"):
                    result = compare_one(eval_outputs, model, method, dataset, metric, n_resamples, seed)
                    if result is None:
                        skipped.append((model, method, dataset, metric))
                    else:
                        rows.append(result)

    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm_correction(df["p_value"].to_numpy())
        df = df.sort_values(["metric", "dataset", "model", "method"]).reset_index(drop=True)
    return df, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                         default=str(Path(__file__).resolve().parent / "origin_vs_finetuned_stats.csv"))
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, skipped = run_all(EVAL_OUTPUTS, args.n_resamples, args.seed)

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} combinations with no completed eval data yet:")
        for s in skipped:
            print(f"  {s}")

    if not len(df):
        print("[INFO] No completed paired_results.csv files found anywhere -- "
              "nothing to analyze yet. Run the eval jobs first.")
        return

    df.to_csv(args.out, index=False)
    print(f"\n[INFO] Wrote {len(df)} comparisons to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""
SPIN vs. {Origin, FFT, LoRA, DoRA} directly (Delta = SPIN - baseline), on
both safety (MD-Judge) and helpfulness (Prometheus-2) metrics, for each
model and each held-out test set (safety_test.csv, helpfulness_test.csv).

Why this script exists alongside stats_origin_vs_finetuned.py: that script
tests each method against origin independently ("does fine-tuning change
behavior at all") -- a different claim from "is SPIN significantly
different from every alternative," which is the paper's actual research
question since SPIN is the method under study (same reasoning already
applied to Table 1's fairness/privacy experiment, where
item_level_stats.py plays this same role next to
item_level_vs_origin_stats.py). Holm correction here is applied across all
24 comparisons in THIS family (SPIN vs. 4 baselines x 3 models x 2
metrics x 1 dataset per run) -- do not mix p-values from this script with
stats_origin_vs_finetuned.py's in one correction pass, they answer
different questions and pooling them would muddle both.

Reuses path-resolution and metric-loading from stats_origin_vs_finetuned.py
(method_paired_results_path, origin_paired_results_path, load_metric_series)
rather than reimplementing them, so both scripts always agree on where the
data lives and how labels/scores are mapped to numbers.

Pairing convention (mirrors item_level_stats.py's convention for Table 1):
  - baseline == "origin": Origin is deterministic (temperature=0, no
    training -- confirmed identical across all 3 "seeds" in every existing
    multiseed run), so only seed 0 is real data. SPIN's 3 seeds are each
    paired against that SAME origin seed-0 file.
  - baseline in {fft, lora, dora}: each has 3 genuinely independent training
    seeds, so SPIN seed s is paired against baseline seed s (matched index)
    for s in 0,1,2, then concatenated -- avoids reusing any one baseline run
    more than once.
Within every pair, prompts are identical row-for-row: eval generation always
reads the same deterministically-ordered safety_test.csv / helpfulness_test.csv.

Usage:
    python spin_vs_baselines_stats.py [--out spin_vs_baselines_item_level_stats.csv]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stats_origin_vs_finetuned import (  # noqa: E402
    EVAL_OUTPUTS, MODELS, DATASETS, SEEDS, ORIGIN_SEED,
    origin_paired_results_path, method_paired_results_path, load_metric_series,
)
from stats_utils import paired_permutation_test, paired_bootstrap_ci, holm_correction  # noqa: E402

BASELINES = ["origin", "fft", "lora", "dora"]


def _load(eval_outputs, model, method, dataset, seed, metric):
    path = origin_paired_results_path(eval_outputs, model, dataset) if method == "origin" \
        else method_paired_results_path(eval_outputs, model, method, dataset, seed)
    if not path.exists():
        return None
    return load_metric_series(path, metric)


def pooled_paired_arrays(eval_outputs, model, baseline, dataset, metric):
    spin_parts, base_parts, seeds_used = [], [], []
    for s in SEEDS:
        spin_vals = _load(eval_outputs, model, "spin", dataset, s, metric)
        base_seed = ORIGIN_SEED if baseline == "origin" else s
        base_vals = _load(eval_outputs, model, baseline, dataset, base_seed, metric)
        if spin_vals is None or base_vals is None:
            continue

        n = min(len(spin_vals), len(base_vals))
        if len(spin_vals) != len(base_vals):
            print(f"[WARN] {model}/spin(seed{s})/{dataset} has {len(spin_vals)} rows, "
                  f"{model}/{baseline}(seed{base_seed})/{dataset} has {len(base_vals)} rows -- "
                  f"truncating to first {n}.")
        x, y = spin_vals.iloc[:n].to_numpy(), base_vals.iloc[:n].to_numpy()

        valid = ~(np.isnan(x) | np.isnan(y))
        if not valid.any():
            continue
        spin_parts.append(x[valid])
        base_parts.append(y[valid])
        seeds_used.append(s)

    if not spin_parts:
        return None, None, []
    return np.concatenate(spin_parts), np.concatenate(base_parts), seeds_used


def compare_one(eval_outputs, model, baseline, dataset, metric, n_resamples=10000, seed=0):
    x, y, seeds_used = pooled_paired_arrays(eval_outputs, model, baseline, dataset, metric)
    if x is None or len(x) < 2:
        return None

    # spin - baseline: positive = SPIN scores higher.
    perm = paired_permutation_test(x, y, n_resamples=n_resamples, seed=seed)
    ci = paired_bootstrap_ci(x, y, n_resamples=n_resamples, seed=seed)
    diffs = x - y

    return {
        "model": model,
        "dataset": dataset,
        "metric": metric,
        "baseline": baseline,
        "n": len(x),
        "seeds_used": ",".join(str(s) for s in seeds_used),
        "spin_mean": float(np.mean(x)),
        "baseline_mean": float(np.mean(y)),
        "mean_delta": float(np.mean(diffs)),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "p_value": perm["p_value"],
        "n_resamples": n_resamples,
    }


def run_all(eval_outputs, n_resamples=10000, seed=0):
    rows, skipped = [], []
    for model in MODELS:
        for dataset in DATASETS:
            for metric in ("safety", "helpfulness"):
                for baseline in BASELINES:
                    result = compare_one(eval_outputs, model, baseline, dataset, metric, n_resamples, seed)
                    if result is None:
                        skipped.append((model, dataset, metric, baseline))
                    else:
                        rows.append(result)

    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm_correction(df["p_value"].to_numpy())
        df["significant_holm_95"] = df["holm_p"] < 0.05
        df = df.sort_values(["metric", "dataset", "model", "baseline"]).reset_index(drop=True)
    return df, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                         default=str(Path(__file__).resolve().parent / "spin_vs_baselines_item_level_stats.csv"))
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, skipped = run_all(EVAL_OUTPUTS, args.n_resamples, args.seed)

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} combinations with no completed eval data yet:")
        for s in skipped:
            print(f"  {s}")

    if not len(df):
        print("[INFO] No completed paired_results.csv files found anywhere.")
        return

    df.to_csv(args.out, index=False)
    print(f"\n[INFO] Wrote {len(df)} comparisons to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Item-level (per-prompt) paired significance test for Table 1: SPIN vs. each
baseline (Origin/FFT/LoRA/DoRA), on both fairness and privacy, for all three
models.

Why this exists: aggregate_multiseed_results.py's significance test pairs by
SEED INDEX (n=3 per cell), which caps the paired permutation test at a
p-value floor of 2/2^3 = 0.25 -- no effect size can ever reach significance
at that n. This script instead pairs by PROMPT (thousands of items per
cell), using the exact same primitives as the safety-helpfulness extension's
stats_origin_vs_finetuned.py: paired_permutation_test for the p-value,
paired_bootstrap_ci for the CI, holm_correction across the whole family.

Requires eval_salad.py's --labels_out_dir per-prompt label CSVs (added
alongside this script -- see eval_origin.py/main.py's --labels_out_dir flag
and the *_multiseed.job files under src/jobs/spin_experiments/multiseed,
reproduction/FFT/finetuning/jobs/multiseed, and
reproduction/LoRA-DoRA/FT-safe-datasets/multiseed), written to:

    item_level_labels/<method>_<model>/seed_<seed>/<category>_labels.csv

with columns: prompt, model_response, label, category. label: 0=aware/safe,
1=unaware/unsafe, -1=judge did not follow the expected format (dropped).

Pairing convention (see conversation/PLAN.md for the reasoning):
  - baseline == "origin": Origin is deterministic (no training, temperature=0
    generation -- confirmed identical across all 3 "seeds" in every existing
    multiseed run), so only its seed-0 labels are real data. SPIN's 3 seeds
    are each paired against that SAME origin seed-0 file (matches
    stats_origin_vs_finetuned.py's ORIGIN_SEED=0 convention exactly).
  - baseline in {fft, lora, dora}: each has 3 genuinely independent training
    seeds, so SPIN seed s is paired against baseline seed s (matched seed
    index) for s in 0,1,2, then concatenated -- avoids reusing any one
    baseline run more than once.
Within every pair, the prompts are identical row-for-row: load_salad_dataset
never shuffles data/salad_{category}.csv, so row i is the same prompt in
every run regardless of method/model/seed.

Usage:
    python item_level_stats.py [--labels_root item_level_labels] [--out item_level_stats.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "n1_localization_controls" / "analysis"))
from stats_utils import paired_permutation_test, paired_bootstrap_ci, holm_correction  # noqa: E402

MODELS = ["qwen", "mistral", "vicuna"]
BASELINES = ["origin", "fft", "lora", "dora"]
CATEGORIES = ["fairness", "privacy"]
SEEDS = [0, 1, 2]


def load_labels(labels_root: Path, method: str, model: str, seed: int, category: str) -> Optional[np.ndarray]:
    """Per-prompt is_aware array (1.0=aware/safe, 0.0=unaware/unsafe, nan=judge
    didn't follow format), in the fixed salad_{category}.csv row order. None
    if the file doesn't exist yet."""
    path = labels_root / f"{method}_{model}" / f"seed_{seed}" / f"{category}_labels.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    label = df["label"].to_numpy(dtype=float)
    is_aware = np.where(label == 0, 1.0, np.where(label == 1, 0.0, np.nan))
    return is_aware


def pooled_paired_arrays(labels_root: Path, model: str, baseline: str, category: str):
    """Concatenate SPIN-vs-baseline per-prompt arrays across SPIN's 3 seeds
    (see module docstring for the baseline=="origin" special case)."""
    spin_parts, base_parts, seeds_used = [], [], []
    for s in SEEDS:
        spin_vals = load_labels(labels_root, "spin", model, s, category)
        base_seed = 0 if baseline == "origin" else s
        base_vals = load_labels(labels_root, baseline, model, base_seed, category)
        if spin_vals is None or base_vals is None:
            continue
        if len(spin_vals) != len(base_vals):
            print(f"[WARN] length mismatch spin/{model}/seed{s} vs {baseline}/{model}/seed{base_seed} "
                  f"({len(spin_vals)} vs {len(base_vals)}) -- skipped")
            continue
        mask = ~(np.isnan(spin_vals) | np.isnan(base_vals))
        if not mask.any():
            continue
        spin_parts.append(spin_vals[mask])
        base_parts.append(base_vals[mask])
        seeds_used.append(s)

    if not spin_parts:
        return None, None, []
    return np.concatenate(spin_parts), np.concatenate(base_parts), seeds_used


def compare_one(labels_root: Path, model: str, baseline: str, category: str, n_resamples: int, seed: int):
    x, y, seeds_used = pooled_paired_arrays(labels_root, model, baseline, category)
    if x is None or len(x) < 2:
        return None

    # spin - baseline: positive = SPIN scores higher.
    perm = paired_permutation_test(x, y, n_resamples=n_resamples, seed=seed)
    ci = paired_bootstrap_ci(x, y, n_resamples=n_resamples, seed=seed)
    diffs = x - y

    return {
        "model": model,
        "metric": category,
        "baseline": baseline,
        "n_items": len(x),
        "seeds_used": ",".join(str(s) for s in seeds_used),
        "spin_mean": float(np.mean(x)),
        "baseline_mean": float(np.mean(y)),
        "mean_delta": float(np.mean(diffs)),
        "ci_lo": ci["ci_low"],
        "ci_hi": ci["ci_high"],
        "p_value": perm["p_value"],
    }


def run_all(labels_root: Path, n_resamples: int = 10000, seed: int = 0):
    rows, skipped = [], []
    for model in MODELS:
        for category in CATEGORIES:
            for baseline in BASELINES:
                result = compare_one(labels_root, model, baseline, category, n_resamples, seed)
                if result is None:
                    skipped.append((model, category, baseline))
                else:
                    rows.append(result)

    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm_correction(df["p_value"].to_numpy())
        df["significant_holm_95"] = df["holm_p"] < 0.05
        df = df.sort_values(["metric", "model", "baseline"]).reset_index(drop=True)
    return df, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Item-level SPIN-vs-baseline significance for Table 1.")
    parser.add_argument("--labels_root", type=Path, default=Path(__file__).with_name("item_level_labels"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("item_level_stats.csv"))
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, skipped = run_all(args.labels_root, args.n_resamples, args.seed)

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} comparisons with no label files yet:")
        for s in skipped:
            print(f"  {s}")

    if not len(df):
        print("[INFO] No label files found -- run the *_multiseed.job jobs with "
              "--labels_out_dir set first.")
        return

    df.to_csv(args.out, index=False)
    print(f"\n[INFO] Wrote {len(df)} comparisons to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

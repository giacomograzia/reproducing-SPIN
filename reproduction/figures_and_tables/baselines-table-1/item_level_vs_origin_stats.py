#!/usr/bin/env python3
"""
Item-level (per-prompt) Delta = method - origin, for EVERY method (FFT, LoRA,
DoRA, SPIN), for Table 1 (fairness/privacy) -- mirrors the safety-helpfulness
extension's stats_origin_vs_finetuned.py exactly (same primitives, same
ORIGIN_SEED=0 convention), but reads item_level_labels/ instead of
paired_results.csv, and reuses item_level_stats.py's load_labels().

Why a separate script from item_level_stats.py: that script computes
SPIN-vs-baseline (SPIN always on one side). This script computes
method-vs-origin for every method, matching the extension table's format
(one row per method, Delta relative to origin) rather than the "does SPIN
beat X" format.

Usage:
    python item_level_vs_origin_stats.py [--labels_root item_level_labels] [--out item_level_vs_origin_stats.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "n1_localization_controls" / "analysis"))
from stats_utils import paired_permutation_test, paired_bootstrap_ci, holm_correction  # noqa: E402

from item_level_stats import load_labels, MODELS, CATEGORIES, SEEDS  # noqa: E402

METHODS = ["fft", "lora", "dora", "spin"]
ORIGIN_SEED = 0  # Origin is deterministic (confirmed identical across all 3 "seeds"); only seed 0 is real data.


def pooled_paired_arrays(labels_root: Path, model: str, method: str, category: str):
    method_parts, origin_parts, seeds_used = [], [], []
    origin_vals = load_labels(labels_root, "origin", model, ORIGIN_SEED, category)
    if origin_vals is None:
        return None, None, []
    for s in SEEDS:
        method_vals = load_labels(labels_root, method, model, s, category)
        if method_vals is None:
            continue
        if len(method_vals) != len(origin_vals):
            print(f"[WARN] length mismatch {method}/{model}/seed{s} vs origin/{model}/seed{ORIGIN_SEED} "
                  f"({len(method_vals)} vs {len(origin_vals)}) -- skipped")
            continue
        mask = ~(np.isnan(method_vals) | np.isnan(origin_vals))
        if not mask.any():
            continue
        method_parts.append(method_vals[mask])
        origin_parts.append(origin_vals[mask])
        seeds_used.append(s)

    if not method_parts:
        return None, None, []
    return np.concatenate(method_parts), np.concatenate(origin_parts), seeds_used


def compare_one(labels_root: Path, model: str, method: str, category: str, n_resamples: int, seed: int):
    y, x, seeds_used = pooled_paired_arrays(labels_root, model, method, category)  # y=method, x=origin
    if y is None or len(y) < 2:
        return None

    perm = paired_permutation_test(y, x, n_resamples=n_resamples, seed=seed)
    ci = paired_bootstrap_ci(y, x, n_resamples=n_resamples, seed=seed)
    diffs = y - x

    return {
        "model": model,
        "metric": category,
        "method": method,
        "n_items": len(y),
        "seeds_used": ",".join(str(s) for s in seeds_used),
        "method_mean": float(np.mean(y)),
        "origin_mean": float(np.mean(x)),
        "mean_delta": float(np.mean(diffs)),
        "ci_lo": ci["ci_low"],
        "ci_hi": ci["ci_high"],
        "p_value": perm["p_value"],
    }


def run_all(labels_root: Path, n_resamples: int = 10000, seed: int = 0):
    rows, skipped = [], []
    for model in MODELS:
        for category in CATEGORIES:
            for method in METHODS:
                result = compare_one(labels_root, model, method, category, n_resamples, seed)
                if result is None:
                    skipped.append((model, category, method))
                else:
                    rows.append(result)

    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm_correction(df["p_value"].to_numpy())
        df["significant_holm_95"] = df["holm_p"] < 0.05
        df = df.sort_values(["metric", "model", "method"]).reset_index(drop=True)
    return df, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Item-level method-vs-origin significance for Table 1.")
    parser.add_argument("--labels_root", type=Path, default=Path(__file__).with_name("item_level_labels"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("item_level_vs_origin_stats.csv"))
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, skipped = run_all(args.labels_root, args.n_resamples, args.seed)

    if skipped:
        print(f"[INFO] Skipped {len(skipped)} comparisons with no label files yet:")
        for s in skipped:
            print(f"  {s}")

    if not len(df):
        print("[INFO] No label files found.")
        return

    df.to_csv(args.out, index=False)
    print(f"\n[INFO] Wrote {len(df)} comparisons to {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

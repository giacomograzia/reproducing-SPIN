#!/usr/bin/env python3
"""
Aggregate multiseed eval logs into mean/std, bootstrap CIs, and paired
significance tests (SPIN vs. each baseline), for Table 1.

Reads the exact "trustworthy results here" block every eval run prints
(eval_origin.py / main.py):

    ===================== trustworthy results here =====================
    {'fairness': {..., 'awareness_rate': np.float64(0.4924)}, 'privacy': {...}}

SPIN-vs-baseline significance now uses the SAME primitives as the
safety-helpfulness extension's stats_origin_vs_finetuned.py -- imported from
n1_localization_controls/analysis/stats_utils.py rather than reimplemented
here -- instead of this file's earlier ad-hoc bootstrap-resample p-value:
paired_permutation_test (a proper permutation/randomization test, the
standard tool for a paired-comparison p-value -- see Dror et al. 2018) for
the p-value, paired_bootstrap_ci (percentile bootstrap) for the CI, and
holm_correction across the whole family of SPIN-vs-baseline comparisons.
Pairing is still by SEED INDEX (n=3 per cell), not by prompt like the
extension -- eval_salad.py's eval_mdjudge() never persists per-example
labels to disk (only the aggregate awareness_rate), so prompt-level pairing
for this experiment would require re-running every eval job with
label-saving added, not just a code change. See PLAN.md / conversation
history for that tradeoff; this file only fixes the statistical procedure on
the seed-level data that already exists.

Usage:
    python aggregate_multiseed_results.py --seeds 0 1 2 --out results_multiseed.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src" / "n1_localization_controls" / "analysis"))
from stats_utils import paired_permutation_test, paired_bootstrap_ci, holm_correction  # noqa: E402

HOME = Path(os.environ.get("HOME", str(Path.home())))
SCRATCH_SHARED = Path(os.environ.get("SCRATCH_SHARED", f"/scratch-shared/{os.environ.get('USER', '')}"))
PROJECT = HOME / "fact-ai-group-23"

# Regex over the single-line dict `main.py`/`eval_origin.py` print at the end.
# `[^}]*` is safe here because each category's inner dict has no nested braces.
# The `np.float64(...)` wrapper is optional: older numpy reprs print bare
# floats (e.g. 'awareness_rate': 0.7159), which is what every multiseed log
# on disk actually contains -- the wrapper-only regex silently matched zero
# logs (all "missing/unparsed") until this was made optional.
RATE_RE = {
    "fairness": re.compile(r"'fairness':\s*\{[^}]*'awareness_rate':\s*(?:np\.float64\()?([0-9.eE+-]+)\)?"),
    "privacy": re.compile(r"'privacy':\s*\{[^}]*'awareness_rate':\s*(?:np\.float64\()?([0-9.eE+-]+)\)?"),
}

METHODS = ["Origin", "FFT", "Lora", "Dora", "Spin"]
MODELS = ["Qwen", "Mistral", "Vicuna"]
METRICS = ["fairness", "privacy"]


def extract_rates(log_path: Path) -> Optional[Dict[str, float]]:
    """Parse one eval log for fairness/privacy awareness_rate. None if missing/unparsed."""
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    out = {}
    for metric, pattern in RATE_RE.items():
        m = pattern.search(text)
        if not m:
            return None
        out[metric] = float(m.group(1))
    return out


def log_path_for(method: str, model: str, seed: int) -> Path:
    """
    Where each multiseed job writes its per-seed eval log, matching the
    LOGDIR/METRICS_ROOT conventions baked into the *_multiseed.job files.
    """
    model_key = {"Qwen": "qwen", "Mistral": "mistral", "Vicuna": "vicuna"}[model]
    model_label = {
        "Qwen": "qwen2-7b-instruct",
        "Mistral": "mistral-7b-instruct-v0.2",
        "Vicuna": "vicuna-7b-v1.5",
    }[model]

    if method == "Origin":
        return PROJECT / "src" / "multiseed_logs" / f"origin_{model_key}" / f"origin_eval_seed{seed}.log"

    if method == "Spin":
        return PROJECT / "src" / "multiseed_logs" / f"spin_{model_key}" / f"spin_eval_seed{seed}.log"

    if method == "FFT":
        return PROJECT / "reproduction" / "FFT" / "finetuning" / "logs" / f"fft_eval_{model_key}_seed{seed}.log"

    if method in ("Lora", "Dora"):
        method_tag = method.lower()
        # Must match the `declare -A MODEL_PATH` keys in the LoRA/DoRA multiseed
        # job files exactly: qwen2_7b, mistral_7b, vicuna_7b (note: no "2" for
        # Mistral). train_model()'s tag is f"{model_key}_{method}_safe_seed{SEED}".
        job_model_key = {"Qwen": "qwen2_7b", "Mistral": "mistral_7b", "Vicuna": "vicuna_7b"}[model]
        if model == "Vicuna":
            saves_root = SCRATCH_SHARED / "fact-ai-group-23-runs" / "llamafactory_saves_safe_vicuna_multiseed" / f"seed_{seed}"
        else:
            saves_root = SCRATCH_SHARED / "fact-ai-group-23-runs" / "llamafactory_saves_safe_multiseed" / f"seed_{seed}"
        tag = f"{job_model_key}_{method_tag}_safe_seed{seed}"
        return saves_root / "metrics" / f"{tag}_merged" / "eval.out"

    raise ValueError(method)


def collect(seeds: List[int]) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """{method: {model: {metric: [values across available seeds]}}}"""
    data: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {mo: {me: [] for me in METRICS} for mo in MODELS} for m in METHODS
    }
    missing = []
    for method in METHODS:
        for model in MODELS:
            for seed in seeds:
                path = log_path_for(method, model, seed)
                rates = extract_rates(path)
                if rates is None:
                    missing.append(f"{method}/{model}/seed={seed}  ({path})")
                    continue
                for metric in METRICS:
                    data[method][model][metric].append(rates[metric])
    if missing:
        print(f"[WARN] {len(missing)} missing/unparsed logs (skipped):")
        for m in missing:
            print(f"  - {m}")
    return data


def bootstrap_ci(values: List[float], n_resamples: int = 10000, alpha: float = 0.05, rng=None) -> Optional[tuple]:
    """Percentile bootstrap CI on the mean. None if fewer than 2 values."""
    if len(values) < 2:
        return None
    rng = rng or np.random.default_rng(12345)
    arr = np.asarray(values)
    resampled_means = rng.choice(arr, size=(n_resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(resampled_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_significance_test(a: List[float], b: List[float], n_resamples: int = 10000, seed: int = 0) -> Optional[dict]:
    """
    SPIN (a) vs. baseline (b), paired by seed index -- same primitives and
    same defaults (n_resamples=10000, seed=0) as the safety-helpfulness
    extension's stats_origin_vs_finetuned.py: paired_permutation_test for the
    p-value (the standard tool for a paired-comparison p-value; a bootstrap
    resample's crossing-zero fraction, used here previously, resamples under
    the observed/alternative distribution rather than the null and is not
    the standard choice) and paired_bootstrap_ci for the CI. holm_correction
    is applied across the whole family of comparisons by the caller, not
    per-cell here.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    perm = paired_permutation_test(a, b, n_resamples=n_resamples, seed=seed)
    ci = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
    return {
        "mean_delta": perm["observed_diff_mean"],
        "ci_lo": ci["ci_low"],
        "ci_hi": ci["ci_high"],
        "p_value": perm["p_value"],
        "n_seeds": len(a),
    }


def format_report(data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> str:
    lines = []
    for method in METHODS:
        for model in MODELS:
            cells = []
            for metric in METRICS:
                vals = data[method][model][metric]
                if not vals:
                    cells.append(f"{metric}=NA")
                    continue
                mean = np.mean(vals)
                std = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                ci = bootstrap_ci(vals)
                ci_str = f", CI95=[{ci[0]:.4f},{ci[1]:.4f}]" if ci else ""
                cells.append(f"{metric}={mean:.4f}±{std:.4f} (n={len(vals)}{ci_str})")
            lines.append(f"{method:8s} {model:8s} " + "  ".join(cells))
    return "\n".join(lines)


def compute_significance_table(
    data: Dict[str, Dict[str, Dict[str, List[float]]]], n_resamples: int = 10000, seed: int = 0
):
    """SPIN vs. every baseline, all (model, metric) cells, one Holm correction
    across the whole family (mirrors stats_origin_vs_finetuned.py's run_all:
    holm_correction is computed once over every p-value in the table, not
    separately per model/metric subgroup)."""
    import pandas as pd

    baselines = [m for m in METHODS if m != "Spin"]
    rows, skipped = [], []
    for model in MODELS:
        for metric in METRICS:
            spin_vals = data["Spin"][model][metric]
            for baseline in baselines:
                base_vals = data[baseline][model][metric]
                result = paired_significance_test(spin_vals, base_vals, n_resamples=n_resamples, seed=seed)
                if result is None:
                    skipped.append((model, metric, baseline))
                    continue
                rows.append({
                    "model": model,
                    "metric": metric,
                    "baseline": baseline,
                    "spin_mean": float(np.mean(spin_vals)),
                    f"{baseline.lower()}_mean": float(np.mean(base_vals)),
                    "mean_delta": result["mean_delta"],
                    "ci_lo": result["ci_lo"],
                    "ci_hi": result["ci_hi"],
                    "p_value": result["p_value"],
                    "n_seeds": result["n_seeds"],
                })

    df = pd.DataFrame(rows)
    if len(df):
        df["holm_p"] = holm_correction(df["p_value"].to_numpy())
        df["significant_holm_95"] = df["holm_p"] < 0.05
        df = df.sort_values(["metric", "model", "baseline"]).reset_index(drop=True)
    return df, skipped


def format_significance(df) -> str:
    lines = []
    for _, r in df.iterrows():
        sig = "SIGNIFICANT" if r["significant_holm_95"] else "not significant"
        lines.append(
            f"{r['model']:8s} {r['metric']:8s} Spin vs {r['baseline']:6s}  "
            f"Δmean={r['mean_delta']:+.4f}  CI95=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
            f"p={r['p_value']:.4f}  holm_p={r['holm_p']:.4f}  {sig}"
        )
    return "\n".join(lines)


def write_python_module(data, path: Path) -> None:
    with path.open("w") as f:
        f.write('"""Auto-generated by aggregate_multiseed_results.py. Do not hand-edit."""\n\n')
        f.write("REPRODUCTION_MULTISEED = {\n")
        for method in METHODS:
            f.write(f"    {method!r}: {{\n")
            for model in MODELS:
                f.write(f"        {model!r}: {{\n")
                for metric in METRICS:
                    vals = data[method][model][metric]
                    mean = float(np.mean(vals)) if vals else None
                    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else None)
                    ci = bootstrap_ci(vals) if vals else None
                    f.write(
                        f"            {metric!r}: {{'values': {vals!r}, 'mean': {mean!r}, "
                        f"'std': {std!r}, 'ci95': {ci!r}}},\n"
                    )
                f.write("        },\n")
            f.write("    },\n")
        f.write("}\n")
    print(f"Wrote: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multiseed Table 1 eval logs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_name("results_multiseed.py"),
        help="Output Python module with aggregated mean/std/CI.",
    )
    parser.add_argument(
        "--stats_out",
        type=Path,
        default=Path(__file__).with_name("spin_vs_baselines_stats.csv"),
        help="Output CSV for the SPIN-vs-baseline significance table.",
    )
    parser.add_argument("--n_resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = collect(args.seeds)

    print("\n===================== MEAN ± STD (bootstrap 95% CI) =====================")
    print(format_report(data))

    sig_df, skipped = compute_significance_table(data, n_resamples=args.n_resamples, seed=args.seed)

    print("\n===================== SPIN vs BASELINES (paired permutation p, bootstrap CI, Holm-corrected) =====================")
    if skipped:
        print(f"[INFO] Skipped {len(skipped)} comparisons with insufficient paired seeds:")
        for s in skipped:
            print(f"  {s}")
    if len(sig_df):
        print(format_significance(sig_df))
        sig_df.to_csv(args.stats_out, index=False)
        print(f"\nWrote: {args.stats_out}")

    write_python_module(data, args.out)


if __name__ == "__main__":
    main()

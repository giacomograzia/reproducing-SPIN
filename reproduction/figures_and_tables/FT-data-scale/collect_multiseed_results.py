#!/usr/bin/env python3
"""
Collect and save multi-seed SPIN limited-data results.

Usage:
    python collect_multiseed_results.py --log_dir ~/fact-ai-group-23/src/multiseed_logs

Reads all spin_eval_${N}_seed${S}.log files, extracts awareness rates,
saves the full results to JSON, prints a mean±std table, and prints
a Python snippet ready to paste into results_dict.py.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path


def parse_trust_results(log_text):
    """Extract the trust_results dict from one log file's text."""
    # The format printed by main.py:
    #   ===================== trustworthy results here =====================
    #   {'fairness': {..., 'awareness_rate': np.float64(0.7852)}, 'privacy': {...}}
    m = re.search(
        r"trustworthy results here\s*=+\s*\n(\{.+?\})\s*\n",
        log_text,
        re.DOTALL,
    )
    if not m:
        return None
    raw = m.group(1).strip()
    # ast.literal_eval can't handle np.float64(...); strip it
    raw = re.sub(r'np\.float\d+\(([^)]+)\)', r'\1', raw)
    try:
        return ast.literal_eval(raw)
    except Exception as e:
        print(f"  [warn] parse error: {e}\n  raw: {raw[:200]}", file=sys.stderr)
        return None


def collect(log_dir: Path):
    """
    Returns a nested dict:
      results[size][seed] = {'fairness': {'awareness_rate': float, ...},
                              'privacy':  {'awareness_rate': float, ...}}
    """
    results = {}
    pattern = re.compile(r"spin_eval_(\d+)_seed(\d+)\.log$")

    for log_file in sorted(log_dir.glob("spin_eval_*_seed*.log")):
        m = pattern.match(log_file.name)
        if not m:
            continue
        size, seed = int(m.group(1)), int(m.group(2))
        parsed = parse_trust_results(log_file.read_text())
        if parsed is None:
            print(f"[skip] could not parse {log_file.name}", file=sys.stderr)
            continue
        results.setdefault(size, {})[seed] = parsed

    return results


def summarise(results):
    """Print a mean±std table and return per-size stats."""
    import numpy as np

    stats = {}
    sizes = sorted(results.keys())
    print(f"\n{'N':>6}  {'fairness mean':>14} {'fairness std':>13}  {'privacy mean':>13} {'privacy std':>12}  seeds")
    print("-" * 75)
    for s in sizes:
        seed_data = results[s]
        f_rates = [v['fairness']['awareness_rate'] for v in seed_data.values()]
        p_rates = [v['privacy']['awareness_rate'] for v in seed_data.values()]
        f_mean, f_std = np.mean(f_rates), np.std(f_rates)
        p_mean, p_std = np.mean(p_rates), np.std(p_rates)
        print(f"{s:>6}  {f_mean:>14.4f} {f_std:>13.4f}  {p_mean:>13.4f} {p_std:>12.4f}  {sorted(seed_data.keys())}")
        stats[s] = {'fairness': {'mean': f_mean, 'std': f_std},
                    'privacy':  {'mean': p_mean,  'std': p_std}}
    return stats


def print_results_dict_snippet(results):
    """Print a Python dict literal ready to paste into SPIN_MULTISEED_RESULTS."""
    sizes = sorted(results.keys())
    print("\n# ── Paste this into results_dict.py as SPIN_MULTISEED_RESULTS ──")
    print("SPIN_MULTISEED_RESULTS = {")
    for s in sizes:
        seed_data = results[s]
        seeds = sorted(seed_data.keys())
        print(f"    {s}: [")
        for seed in seeds:
            r = seed_data[seed]
            f = r['fairness']['awareness_rate']
            p = r['privacy']['awareness_rate']
            print(f"        {{'fairness': {{'awareness_rate': {f:.4f}}}, 'privacy': {{'awareness_rate': {p:.4f}}}}},  # seed {seed}")
        print("    ],")
    print("}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path.home() / "fact-ai-group-23/src/multiseed_logs",
        help="Directory containing spin_eval_${N}_seed${S}.log files",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Save full per-seed results to this JSON file (default: <log_dir>/multiseed_results.json)",
    )
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser()
    if not log_dir.exists():
        sys.exit(f"Log directory not found: {log_dir}")

    results = collect(log_dir)
    if not results:
        sys.exit("No log files found matching spin_eval_<N>_seed<S>.log")

    # Save full results
    out_json = args.out_json or log_dir / "multiseed_results.json"
    # JSON can't store non-string keys; cast to str
    json_safe = {str(size): {str(seed): data for seed, data in seeds.items()}
                 for size, seeds in results.items()}
    out_json.write_text(json.dumps(json_safe, indent=2))
    print(f"Full results saved to {out_json}")

    # Print summary table
    summarise(results)

    # Print snippet
    print_results_dict_snippet(results)


if __name__ == "__main__":
    main()

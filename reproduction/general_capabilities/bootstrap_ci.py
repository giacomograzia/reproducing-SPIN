#!/usr/bin/env python3
"""
Paired item-level bootstrap CIs for the general-capabilities benchmarks
(HellaSwag, MMLU, WinoGrande, perplexity).

WHY BOOTSTRAP, NOT SEEDS: eval_v2.py's decoding is greedy (do_sample=False,
temperature=0.0) -- MCQ/bool answers are the argmax of a single generated
token -- and perplexity is one fixed forward pass over a fixed wikitext
segmentation. Neither path consumes any RNG state (no dropout at eval time,
no sampling, deepeval's HellaSwag/MMLU/Winogrande use their full fixed eval
split with deterministic few-shot selection). Re-running with a different
model/eval seed therefore reproduces an identical score -- seed-based error
bars would be meaningless here.

The actual source of uncertainty is which *items* happen to be in the fixed
test set: resampling item indices (with replacement) and recomputing the
score under each resample estimates how much the metric -- and the
Origin-vs-other-method difference -- would wobble under a different draw of
test items from the same underlying distribution. Using the SAME resampled
indices for both models in every bootstrap iteration (a *paired* bootstrap)
is what makes the difference's CI valid: item-level noise that hits both
models' scores in the same direction cancels out of the difference, so the
diff CI reflects genuine disagreement between the models, not shared item
variance. This runs in seconds against the per-item predictions eval_v2.py
already saved -- no additional model forward passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PER_ITEM_DIR = Path(__file__).resolve().parent / "per_item"
# Over-refusal benchmarks (written by score_refusal.py) are accuracy-shaped by
# construction: one 0/1 per item over a fixed prompt set in fixed order, with
# `correct` defined higher-is-better (compliance on benign prompts, refusal on
# harmful ones). They therefore need no new bootstrap machinery -- the same
# paired resampling applies unchanged.
REFUSAL_BENCHMARKS = ("xstest_safe", "xstest_unsafe", "orbench_safe")
ACCURACY_BENCHMARKS = ("hellaswag", "mmlu", "winogrande") + REFUSAL_BENCHMARKS
ALL_BENCHMARKS = ACCURACY_BENCHMARKS + ("perplexity",)


def load_per_item(benchmark: str, model_id: str) -> pd.DataFrame:
    path = PER_ITEM_DIR / f"{benchmark}__{model_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No per-item file for benchmark={benchmark!r} model={model_id!r}: {path}")
    return pd.read_parquet(path)


def align_or_die(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str) -> None:
    """Hard-fail on any N or item_id mismatch -- the pairing is the whole point:
    silently re-aligning (e.g. via a merge) would let a bug in one model's
    item ordering/coverage quietly produce a wrong CI instead of an error."""
    if len(df_a) != len(df_b):
        raise AssertionError(f"N mismatch: {name_a} has {len(df_a)} items, {name_b} has {len(df_b)}")
    ids_a = df_a["item_id"].to_numpy()
    ids_b = df_b["item_id"].to_numpy()
    if not np.array_equal(ids_a, ids_b):
        raise AssertionError(
            f"item_id mismatch between {name_a} and {name_b}: pairing requires identical "
            "item_ids in identical order (same benchmark construction for both models)."
        )


def percentile_ci(values: np.ndarray, lo: float = 2.5, hi: float = 97.5) -> tuple:
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def bootstrap_two_sided_p(diff_boot: np.ndarray) -> float:
    p = 2 * min(float((diff_boot <= 0).mean()), float((diff_boot >= 0).mean()))
    return float(min(p, 1.0))


def paired_bootstrap_accuracy(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int, rng: np.random.Generator):
    n = len(correct_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = correct_a[idx].mean(axis=1)
    boot_b = correct_b[idx].mean(axis=1)
    return boot_a, boot_b


def paired_bootstrap_perplexity(
    nll_a: np.ndarray, tok_a: np.ndarray, nll_b: np.ndarray, tok_b: np.ndarray, n_boot: int, rng: np.random.Generator
):
    n = len(nll_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_a = np.exp(nll_a[idx].sum(axis=1) / tok_a[idx].sum(axis=1))
    boot_b = np.exp(nll_b[idx].sum(axis=1) / tok_b[idx].sum(axis=1))
    return boot_a, boot_b


def mcnemar_exact_p(correct_a: np.ndarray, correct_b: np.ndarray) -> Optional[float]:
    """Two-sided exact McNemar p-value for two paired binary vectors.

    An analytic cross-check on the bootstrap for the strictly-paired benchmarks:
    same items, two models, binary outcome. Only the discordant pairs carry
    information (b = A right/B wrong, c = A wrong/B right), and under the null
    each discordant pair is a fair coin, so the p-value is a two-sided binomial
    test on b out of b+c at p=0.5.

    Computed with an exact binomial sum rather than the chi-square
    approximation, which is unreliable when b+c is small -- entirely possible on
    a 200-item arm where the two models mostly agree. Returns None when there
    are no discordant pairs (the models agree on every item, so there is no
    evidence of a difference either way).
    """
    a = correct_a.astype(int)
    b_count = int(((a == 1) & (correct_b.astype(int) == 0)).sum())
    c_count = int(((a == 0) & (correct_b.astype(int) == 1)).sum())
    n_disc = b_count + c_count
    if n_disc == 0:
        return None

    from math import comb

    k = min(b_count, c_count)
    tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
    return float(min(1.0, 2 * tail))


def compute_for_benchmark(benchmark: str, origin_name: str, spin_name: str, n_boot: int, seed: int) -> dict:
    """seed controls ONLY the resampling RNG (reproducibility of the CI
    itself) -- it has nothing to do with, and should not be confused with,
    any model/training/eval seed."""
    rng = np.random.default_rng(seed)
    df_o = load_per_item(benchmark, origin_name)
    df_s = load_per_item(benchmark, spin_name)
    align_or_die(df_o, df_s, origin_name, spin_name)

    mcnemar_p = None
    if benchmark == "perplexity":
        nll_o = df_o["nll_sum"].to_numpy(dtype=float)
        tok_o = df_o["n_tokens"].to_numpy(dtype=float)
        nll_s = df_s["nll_sum"].to_numpy(dtype=float)
        tok_s = df_s["n_tokens"].to_numpy(dtype=float)

        point_o = float(np.exp(nll_o.sum() / tok_o.sum()))
        point_s = float(np.exp(nll_s.sum() / tok_s.sum()))
        boot_o, boot_s = paired_bootstrap_perplexity(nll_o, tok_o, nll_s, tok_s, n_boot, rng)
    else:
        c_o = df_o["correct"].to_numpy(dtype=float)
        c_s = df_s["correct"].to_numpy(dtype=float)

        point_o = float(c_o.mean())
        point_s = float(c_s.mean())
        boot_o, boot_s = paired_bootstrap_accuracy(c_o, c_s, n_boot, rng)
        mcnemar_p = mcnemar_exact_p(c_o, c_s)

    boot_diff = boot_s - boot_o
    point_diff = point_s - point_o

    return {
        "benchmark": benchmark,
        "n_items": len(df_o),
        "origin_name": origin_name,
        "spin_name": spin_name,
        "origin_score": point_o,
        "origin_ci": percentile_ci(boot_o),
        "spin_score": point_s,
        "spin_ci": percentile_ci(boot_s),
        "diff": point_diff,
        "diff_ci": percentile_ci(boot_diff),
        "p_value": bootstrap_two_sided_p(boot_diff),
        "mcnemar_p": mcnemar_p,
    }


def print_single_result(r: dict) -> None:
    print(f"\n===================== {r['benchmark']} (n={r['n_items']}) =====================")
    print(f"{r['origin_name']:30s} {r['origin_score']:.3f}  CI95=[{r['origin_ci'][0]:.3f}, {r['origin_ci'][1]:.3f}]")
    print(f"{r['spin_name']:30s} {r['spin_score']:.3f}  CI95=[{r['spin_ci'][0]:.3f}, {r['spin_ci'][1]:.3f}]")
    sig = "SIGNIFICANT" if not (r["diff_ci"][0] <= 0 <= r["diff_ci"][1]) else "not significant (CI contains 0)"
    print(
        f"diff (spin - origin): {r['diff']:+.3f}  CI95=[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]  "
        f"p={r['p_value']:.3f}  {sig}"
    )
    if r.get("mcnemar_p") is not None:
        # Analytic cross-check; large disagreement with the bootstrap p would
        # point at a problem with the pairing rather than at a real effect.
        print(f"McNemar exact p={r['mcnemar_p']:.3f} (paired-binary cross-check)")


def selftest() -> bool:
    """Bootstrapping a vector against itself: with the SAME resampled indices
    applied to both (identical) vectors, every bootstrap replicate's diff is
    exactly 0, so the diff CI must collapse to [0, 0] and p must be 1.0.
    This validates the pairing mechanics, not any real data."""
    rng_data = np.random.default_rng(42)
    correct = (rng_data.random(200) < 0.7).astype(float)  # synthetic 0/1 vector

    rng = np.random.default_rng(0)
    boot_a, boot_b = paired_bootstrap_accuracy(correct, correct, n_boot=2000, rng=rng)
    diff_boot = boot_b - boot_a
    ci = percentile_ci(diff_boot)
    p = bootstrap_two_sided_p(diff_boot)

    ok = abs(ci[0]) < 1e-9 and abs(ci[1]) < 1e-9 and abs(p - 1.0) < 1e-9
    print(f"[selftest] diff CI = [{ci[0]:.6f}, {ci[1]:.6f}], p = {p:.6f} -> {'PASS' if ok else 'FAIL'}")
    return ok


def run_all(pairs: list, n_boot: int, seed: int, out_csv: Path) -> None:
    """
    pairs: list of (origin_name, other_name) tuples, e.g.
      [("origin_qwen2-7b-instruct", "spin_qwen2-7b-instruct"),
       ("origin_mistral-7b-instruct-v0.2", "fft_mistral-7b-instruct-v0.2"), ...]
    Explicit pairs only -- no fuzzy name-matching, to avoid silently pairing
    the wrong models (e.g. a Qwen SPIN run against a Mistral Origin run).
    Writes one row per (model, benchmark): the Origin itself (diff_vs_origin=0,
    trivial CI) plus each paired "other" model, for every benchmark in
    ALL_BENCHMARKS that has per-item files for that pair.
    """
    rows = []
    seen_origins = set()
    for origin_name, other_name in pairs:
        for benchmark in ALL_BENCHMARKS:
            origin_path = PER_ITEM_DIR / f"{benchmark}__{origin_name}.parquet"
            other_path = PER_ITEM_DIR / f"{benchmark}__{other_name}.parquet"
            if not origin_path.exists() or not other_path.exists():
                print(f"[WARN] skipping {benchmark} for ({origin_name}, {other_name}): missing per-item file(s)")
                continue

            r = compute_for_benchmark(benchmark, origin_name, other_name, n_boot, seed)

            if (origin_name, benchmark) not in seen_origins:
                seen_origins.add((origin_name, benchmark))
                rows.append({
                    "model": origin_name, "benchmark": benchmark,
                    "score": round(r["origin_score"], 3),
                    "ci_low": round(r["origin_ci"][0], 3), "ci_high": round(r["origin_ci"][1], 3),
                    "diff_vs_origin": 0.0, "diff_ci_low": 0.0, "diff_ci_high": 0.0, "p_value": 1.0,
                    "mcnemar_p": "",
                })

            rows.append({
                "model": other_name, "benchmark": benchmark,
                "score": round(r["spin_score"], 3),
                "ci_low": round(r["spin_ci"][0], 3), "ci_high": round(r["spin_ci"][1], 3),
                "diff_vs_origin": round(r["diff"], 3),
                "diff_ci_low": round(r["diff_ci"][0], 3), "diff_ci_high": round(r["diff_ci"][1], 3),
                "p_value": round(r["p_value"], 3),
                "mcnemar_p": "" if r["mcnemar_p"] is None else round(r["mcnemar_p"], 3),
            })

    if not rows:
        print("[WARN] no rows produced -- check that per-item files exist for the given --pairs")
        return

    df = pd.DataFrame(rows, columns=[
        "model", "benchmark", "score", "ci_low", "ci_high",
        "diff_vs_origin", "diff_ci_low", "diff_ci_high", "p_value", "mcnemar_p",
    ])
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=str, choices=ALL_BENCHMARKS)
    parser.add_argument("--origin_name", type=str, help="model_id for the origin per-item file(s)")
    parser.add_argument("--spin_name", type=str, help="model_id for the other model's per-item file(s)")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0,
                         help="Resampling RNG seed for the bootstrap itself (reproducibility of the "
                              "CI) -- NOT a model/training/eval seed.")
    parser.add_argument("--all", action="store_true",
                         help="Run every benchmark for a set of explicit origin:other pairs and write table2_ci.csv.")
    parser.add_argument("--pairs", nargs="+",
                         help="For --all: one or more 'origin_name:other_name' model_id pairs.")
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("table2_ci.csv"))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if args.all:
        if not args.pairs:
            parser.error("--all requires --pairs origin_name:other_name [origin_name:other_name ...]")
        pairs = []
        for p in args.pairs:
            if ":" not in p:
                parser.error(f"--pairs entries must be 'origin_name:other_name', got: {p!r}")
            o, s = p.split(":", 1)
            pairs.append((o, s))
        run_all(pairs, args.n_boot, args.seed, args.out)
        return

    if not (args.benchmark and args.origin_name and args.spin_name):
        parser.error("single-comparison mode requires --benchmark, --origin_name, and --spin_name")

    r = compute_for_benchmark(args.benchmark, args.origin_name, args.spin_name, args.n_boot, args.seed)
    print_single_result(r)


if __name__ == "__main__":
    main()

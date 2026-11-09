#!/usr/bin/env python3
"""
CPU-only validation of the over-refusal pipeline's non-GPU half.

Exercises everything except vLLM generation: the refusal classifier, the two
prompt-set loaders, the JSONL -> per-item-parquet scorer, and the paired
bootstrap + McNemar path in bootstrap_ci.py. Run this before submitting the GPU
jobs -- every bug it can catch would otherwise surface only at the very END of a
~30 min SPIN job, after the checkpoint re-derivation is already spent.

Uses a temporary directory for generated artifacts so it never writes into the
real per_item/ or refusal_generations/ directories.

    python test_refusal_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src"
for p in (str(SRC), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import bootstrap_ci  # noqa: E402
import score_refusal  # noqa: E402
from eval_refusal import (  # noqa: E402
    COMPLIANCE,
    EMPTY,
    FULL_REFUSAL,
    PARTIAL_REFUSAL,
    classify_refusal,
    load_refusal_dataset,
)

failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


def test_classifier() -> None:
    print("\n=== refusal classifier ===")
    cases = [
        ("I'm sorry, but I can't help with that.", FULL_REFUSAL),
        ("I cannot provide instructions for that.", FULL_REFUSAL),
        ("As an AI language model, I must decline.", FULL_REFUSAL),
        ("Sorry, that's not something I can assist with.", FULL_REFUSAL),
        ("To kill a Python process, use kill -9 <pid>.", COMPLIANCE),
        ("You can shoot a great photo at dusk using a tripod.", COMPLIANCE),
        # Partial: opens by complying, then hedges with refusal language.
        ("Here's some general info. However, I cannot give specifics on that.", PARTIAL_REFUSAL),
        ("", EMPTY),
        ("   ", EMPTY),
        (None, EMPTY),
    ]
    for text, expected in cases:
        got = classify_refusal(text)
        label = repr(text)[:52]
        check(f"classify {label} -> {expected}", got == expected, f"got {got}")


def test_datasets() -> None:
    print("\n=== prompt sets ===")
    x = load_refusal_dataset("xstest")
    check("xstest has 450 rows", len(x) == 450, f"got {len(x)}")
    n_safe = int((x["label"] == "safe").sum())
    n_unsafe = int((x["label"] == "unsafe").sum())
    check("xstest 250 safe / 200 unsafe", (n_safe, n_unsafe) == (250, 200), f"got {n_safe}/{n_unsafe}")
    check("xstest item_ids unique", x["item_id"].is_unique)
    check("xstest unsafe arm is the contrast_* types",
          bool(x[x["label"] == "unsafe"]["group"].str.startswith("contrast_").all()))

    o = load_refusal_dataset("or_bench")
    check("or_bench non-empty", len(o) > 0, f"got {len(o)}")
    check("or_bench all benign", bool((o["label"] == "safe").all()))
    check("or_bench has >=10 categories", o["group"].nunique() >= 10, f"got {o['group'].nunique()}")


def _write_synthetic(gen_dir: Path, model_id: str, refusal_rate_safe: float, seed: int) -> None:
    """Fake generations for XSTest: a model that refuses a given fraction of the
    safe arm and always refuses the unsafe arm."""
    rng = np.random.default_rng(seed)
    prompts = load_refusal_dataset("xstest")
    rows = []
    for r in prompts.to_dict(orient="records"):
        if r["label"] == "unsafe":
            resp = "I cannot help with that."
        else:
            resp = ("I'm sorry, but I can't help with that."
                    if rng.random() < refusal_rate_safe else "Sure! Here is how you do it.")
        rows.append({**r, "response": resp})
    with open(gen_dir / f"xstest__{model_id}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_scoring_and_ci() -> None:
    print("\n=== scorer + paired bootstrap ===")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gen_dir, item_dir, res_dir = tmp / "gen", tmp / "per_item", tmp / "results"
        gen_dir.mkdir()

        # Redirect the module's output paths so the real directories are untouched.
        score_refusal.GEN_DIR = gen_dir
        score_refusal.PER_ITEM_DIR = item_dir
        score_refusal.RESULTS_DIR = res_dir
        bootstrap_ci.PER_ITEM_DIR = item_dir

        _write_synthetic(gen_dir, "origin_fake", refusal_rate_safe=0.10, seed=1)
        _write_synthetic(gen_dir, "spin_fake", refusal_rate_safe=0.30, seed=2)

        for model_id in ("origin_fake", "spin_fake"):
            score_refusal.score_model("xstest", model_id, "full")

        check("per-item files written",
              (item_dir / "xstest_safe__origin_fake.parquet").exists()
              and (item_dir / "xstest_unsafe__spin_fake.parquet").exists())

        # Sign convention: on the safe arm `correct` = complied, so the model
        # that over-refuses more must score LOWER.
        r = bootstrap_ci.compute_for_benchmark("xstest_safe", "origin_fake", "spin_fake", n_boot=2000, seed=0)
        check("safe-arm n=250", r["n_items"] == 250, f"got {r['n_items']}")
        check("over-refusing model scores lower on safe arm",
              r["spin_score"] < r["origin_score"],
              f"origin={r['origin_score']:.3f} spin={r['spin_score']:.3f}")
        check("20pp planted difference is detected",
              r["diff"] < 0 and not (r["diff_ci"][0] <= 0 <= r["diff_ci"][1]),
              f"diff={r['diff']:.3f} CI={r['diff_ci']}")
        check("McNemar agrees with bootstrap (both significant)",
              r["mcnemar_p"] is not None and r["mcnemar_p"] < 0.05 and r["p_value"] < 0.05,
              f"mcnemar={r['mcnemar_p']} boot={r['p_value']}")

        # Unsafe arm: both fakes always refuse, i.e. both score 1.0. The
        # difference must be exactly zero, and McNemar must return None rather
        # than a p-value, since there are no discordant pairs at all.
        r2 = bootstrap_ci.compute_for_benchmark("xstest_unsafe", "origin_fake", "spin_fake", n_boot=500, seed=0)
        check("identical unsafe arms -> zero diff", r2["diff"] == 0.0, f"got {r2['diff']}")
        check("McNemar returns None when no discordant pairs", r2["mcnemar_p"] is None, f"got {r2['mcnemar_p']}")

        # Pairing must be enforced, not silently repaired.
        try:
            bootstrap_ci.compute_for_benchmark("xstest_safe", "origin_fake", "xstest_unsafe_missing", 100, 0)
            check("missing per-item file raises", False, "no exception raised")
        except FileNotFoundError:
            check("missing per-item file raises", True)


def test_mcnemar_math() -> None:
    print("\n=== McNemar exact ===")
    # 10 discordant pairs all in one direction: two-sided exact p = 2 * 0.5^10.
    a = np.array([1] * 10 + [0] * 90)
    b = np.array([0] * 10 + [0] * 90)
    p = bootstrap_ci.mcnemar_exact_p(a, b)
    check("all-discordant-one-way p = 2*0.5^10", abs(p - 2 * 0.5 ** 10) < 1e-12, f"got {p}")

    # Perfectly balanced discordance -> no evidence of a difference.
    a2 = np.array([1] * 5 + [0] * 5)
    b2 = np.array([0] * 5 + [1] * 5)
    check("balanced discordance p = 1.0", abs(bootstrap_ci.mcnemar_exact_p(a2, b2) - 1.0) < 1e-12)


def main() -> int:
    test_classifier()
    test_datasets()
    test_mcnemar_math()
    test_scoring_and_ci()

    print("\n=== bootstrap_ci selftest (unchanged existing behaviour) ===")
    check("bootstrap_ci --selftest still passes", bootstrap_ci.selftest())

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

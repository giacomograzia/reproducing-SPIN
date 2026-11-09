"""
Smoke test for stats_origin_vs_finetuned.py -- hand-constructed paired_results.csv
files (no GPU/model involved) checked against by-hand-derivable expected values.
Not a pytest suite -- run directly:
    python test_stats_origin_vs_finetuned.py
Genuinely brief (sub-second, tiny fabricated CSVs), so per the project's compute
policy this is fine to run directly on the login node, same as
n1_localization_controls/analysis/test_stats_utils.py.

Covers:
  - safety metric: safety_label -> is_safe conversion (0/1 -> 1.0/0.0), and
    that a -1 ("no_following") entry drops that row from the pairing.
  - helpfulness metric: -1 ("invalid_format") entries drop that row.
  - pooling across the method's seeds against a SINGLE reused origin result:
    two of the method's seeds' valid rows get concatenated into one paired
    array pair, both paired against the one origin file (origin has no
    per-seed variation -- see ORIGIN_SEED in stats_origin_vs_finetuned.py).
  - missing files: a (model, method, dataset) combination with no files on
    disk is skipped, not an error.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_origin_vs_finetuned import (  # noqa: E402
    compare_one,
    origin_paired_results_path,
    method_paired_results_path,
    run_all,
)


def _write_paired_results(path, safety_labels=None, helpfulness_scores=None):
    # Default the metric NOT under test to all -1 ("invalid"/"no_following"),
    # not a real value -- otherwise a fixture written for one metric would
    # accidentally also produce a "valid" comparison for the other metric,
    # since -1 is the only value load_metric_series treats as missing.
    n = len(safety_labels) if safety_labels is not None else len(helpfulness_scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "prompt": [f"p{i}" for i in range(n)],
        "model_response": [f"r{i}" for i in range(n)],
        "reference_answer": [f"ref{i}" for i in range(n)],
        "safety_label": safety_labels if safety_labels is not None else [-1] * n,
        "safety_category": [""] * n,
        "helpfulness_score": helpfulness_scores if helpfulness_scores is not None else [-1] * n,
    }).to_csv(path, index=False)


def main():
    tmp_root = Path(tempfile.mkdtemp(prefix="stats_smoketest_"))
    try:
        # ---- safety metric, method has 2 seeds, one -1 entry in seed1's fft
        # row, both paired against the SAME single origin result ----
        # origin: labels [0,0,1] -> is_safe [1,1,0]
        _write_paired_results(
            origin_paired_results_path(tmp_root, "qwen", "safety_test"),
            safety_labels=[0, 0, 1],
        )
        # fft seed0: labels [0,0,0] -> is_safe [1,1,1]
        _write_paired_results(
            method_paired_results_path(tmp_root, "qwen", "fft", "safety_test", 0),
            safety_labels=[0, 0, 0],
        )
        # fft seed1: labels [0,-1,0] -> is_safe [1,NaN,1] -- row 1 must be dropped
        _write_paired_results(
            method_paired_results_path(tmp_root, "qwen", "fft", "safety_test", 1),
            safety_labels=[0, -1, 0],
        )

        result = compare_one(tmp_root, "qwen", "fft", "safety_test", "safety", n_resamples=1000, seed=0)
        assert result is not None, "expected a result, got None"
        assert result["n"] == 5, f"expected 5 pooled valid rows (3 + 2 after dropping the -1), got {result['n']}"
        assert result["seeds_used"] == "0,1"
        expected_mean_diff = np.mean([0, 0, 1, 0, 1])  # y-x per pooled row, see docstring math
        assert abs(result["mean_diff"] - expected_mean_diff) < 1e-9, \
            f"expected mean_diff={expected_mean_diff}, got {result['mean_diff']}"
        assert abs(result["origin_mean"] - np.mean([1, 1, 0, 1, 0])) < 1e-9
        assert abs(result["method_mean"] - np.mean([1, 1, 1, 1, 1])) < 1e-9
        print("[PASS] safety metric: is_safe conversion, -1 dropping, seed pooling all correct")

        # ---- helpfulness metric, single seed, one -1 entry ----
        _write_paired_results(
            origin_paired_results_path(tmp_root, "mistral", "helpfulness_test"),
            helpfulness_scores=[3, 4, 5],
        )
        _write_paired_results(
            method_paired_results_path(tmp_root, "mistral", "fft", "helpfulness_test", 0),
            helpfulness_scores=[4, 5, -1],
        )
        result = compare_one(tmp_root, "mistral", "fft", "helpfulness_test", "helpfulness",
                              n_resamples=1000, seed=0)
        assert result is not None
        assert result["n"] == 2, f"expected 2 valid rows (row 2 dropped: method score is -1), got {result['n']}"
        assert abs(result["mean_diff"] - 1.0) < 1e-9, f"expected mean_diff=1.0, got {result['mean_diff']}"
        assert abs(result["origin_mean"] - 3.5) < 1e-9
        assert abs(result["method_mean"] - 4.5) < 1e-9
        print("[PASS] helpfulness metric: -1 dropping and mean_diff correct")

        # ---- missing files: no data written for vicuna/dora anywhere ----
        result = compare_one(tmp_root, "vicuna", "dora", "safety_test", "safety", n_resamples=1000, seed=0)
        assert result is None, "expected None when no files exist for this combination"
        print("[PASS] missing-file combination correctly returns None (not an error)")

        # ---- run_all: qwen/fft/safety_test/safety should appear in results,
        # everything else should be in skipped (no files written for it) ----
        df, skipped = run_all(tmp_root, n_resamples=1000, seed=0)
        assert len(df) == 2, f"expected exactly 2 completed comparisons (the two written above), got {len(df)}"
        assert ("qwen", "fft", "safety_test", "safety") not in skipped
        assert ("vicuna", "dora", "safety_test", "safety") in skipped
        assert "holm_p" in df.columns
        print(f"[PASS] run_all: {len(df)} comparisons found, {len(skipped)} correctly skipped, holm_p present")

        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()

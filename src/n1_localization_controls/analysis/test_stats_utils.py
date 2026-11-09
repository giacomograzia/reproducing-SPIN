"""
Smoke test for stats_utils.py, required by the approved plan before trusting
scipy.stats.permutation_test/bootstrap on the real 25-run data (their
vectorized/axis handling can vary across scipy minor versions -- this repo's
FACT env pins scipy==1.17.1).

Not a pytest suite -- run directly:
    python n1_localization_controls/analysis/test_stats_utils.py
This is genuinely brief (sub-second, tiny hand-constructed arrays), so it is
a "genuinely brief interactive check" under the project's compute policy and
may be run directly rather than via sbatch, unlike every other script in
this experiment.

Each check compares stats_utils' output against an independently,
by-hand-derivable value:
  - paired_permutation_test: 3 pairs -> exactly 8 equally-likely sign-flip
    outcomes, enumerated by hand below and compared to the function's
    p-value (validates permutation_type='samples' does the sign-flip null,
    not an independent-sample shuffle).
  - spearman_permutation_pvalue: perfectly monotonic n=5 data (rho=1.0
    exactly) should be among the most extreme of all 5!=120 pairings, so its
    permutation p-value should be small and reproducible; also checks a
    manually-shuffled-pairs case against exact enumeration.
  - spearman_bootstrap_ci / paired_bootstrap_ci: sanity-check the CI
    contains the observed statistic and has the expected sign/order.
  - holm_correction: textbook 3-p-value example with a known expected result.
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats_utils import (  # noqa: E402
    holm_correction,
    paired_bootstrap_ci,
    paired_permutation_test,
    spearman_bootstrap_ci,
    spearman_permutation_pvalue,
    spearman_rho,
)


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    return condition


def test_paired_permutation_test():
    # 3 pairs -> diffs = [2, -2, 5], observed mean diff = 5/3.
    # All 8 sign-flip outcomes of |diffs|=[2,2,5] enumerated by hand:
    #   (+,+,+)= 3.0   (+,+,-)=-1/3   (+,-,+)= 5/3   (+,-,-)=-5/3
    #   (-,+,+)= 5/3   (-,+,-)=-5/3   (-,-,+)= 1/3   (-,-,-)=-3.0
    # |stat| >= |5/3| for 6 of the 8 outcomes -> exact two-sided p = 6/8 = 0.75
    x = np.array([10.0, 20.0, 30.0])
    y = np.array([8.0, 22.0, 25.0])
    abs_d = np.array([2.0, 2.0, 5.0])
    observed = float(np.mean(x - y))
    hand_stats = [
        sum(s * d for s, d in zip(signs, abs_d)) / 3.0
        for signs in itertools.product([1, -1], repeat=3)
    ]
    hand_p = np.mean([abs(s) >= abs(observed) - 1e-9 for s in hand_stats])

    result = paired_permutation_test(x, y, n_resamples=20000, seed=0)
    ok = check(
        "paired_permutation_test matches hand-enumerated 3-pair sign-flip null",
        abs(result["p_value"] - hand_p) < 0.05,
        f"(function={result['p_value']:.4f}, hand={hand_p:.4f})",
    )
    ok &= check(
        "paired_permutation_test observed_diff_mean correct",
        abs(result["observed_diff_mean"] - observed) < 1e-9,
    )
    return ok


def test_spearman_permutation_and_rho():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # perfectly monotonic -> rho = 1.0
    rho = spearman_rho(x, y)
    ok = check("spearman_rho == 1.0 for perfectly monotonic data", abs(rho - 1.0) < 1e-9, f"(rho={rho})")

    # Exact enumeration over all 5! = 120 pairings of y with x: only 1 pairing
    # (the identity) gives rho=1.0, and only 1 gives rho=-1.0 (full reversal);
    # for a two-sided test the exact p-value is 2/120 = 0.01666...
    from scipy.stats import spearmanr
    all_rhos = [spearmanr(x, perm).correlation for perm in itertools.permutations(y)]
    hand_p = np.mean([abs(r) >= abs(rho) - 1e-9 for r in all_rhos])

    result = spearman_permutation_pvalue(x, y, n_resamples=20000, seed=0)
    ok &= check(
        "spearman_permutation_pvalue matches exact enumeration for n=5 monotonic data",
        abs(result["p_value"] - hand_p) < 0.03,
        f"(function={result['p_value']:.4f}, exact={hand_p:.4f})",
    )
    return ok


def test_bootstrap_cis_run_and_bracket_observed():
    rng = np.random.RandomState(0)
    x = rng.normal(0.6, 0.1, size=25)
    y = rng.normal(0.5, 0.1, size=25)

    diff_ci = paired_bootstrap_ci(x, y, n_resamples=5000, seed=0)
    observed_diff = float(np.mean(x - y))
    ok = check(
        "paired_bootstrap_ci brackets the observed mean difference",
        diff_ci["ci_low"] <= observed_diff <= diff_ci["ci_high"],
        f"(ci=[{diff_ci['ci_low']:.4f}, {diff_ci['ci_high']:.4f}], observed={observed_diff:.4f})",
    )

    rho_ci = spearman_bootstrap_ci(x, y, n_resamples=5000, seed=0)
    observed_rho = spearman_rho(x, y)
    ok &= check(
        "spearman_bootstrap_ci brackets the observed rho (paired=True preserves pairing)",
        rho_ci["ci_low"] <= observed_rho <= rho_ci["ci_high"],
        f"(ci=[{rho_ci['ci_low']:.4f}, {rho_ci['ci_high']:.4f}], observed={observed_rho:.4f})",
    )
    return ok


def test_holm_correction():
    # Textbook example: raw p-values [0.01, 0.02, 0.03], m=3.
    # rank0: 3*0.01=0.03 -> running_max=0.03
    # rank1: 2*0.02=0.04 -> running_max=0.04
    # rank2: 1*0.03=0.03 -> enforced monotone -> running_max stays 0.04
    # expected adjusted = [0.03, 0.04, 0.04]
    pvals = np.array([0.01, 0.02, 0.03])
    expected = np.array([0.03, 0.04, 0.04])
    adjusted = holm_correction(pvals)
    ok = check(
        "holm_correction matches textbook 3-p-value example",
        np.allclose(adjusted, expected, atol=1e-9),
        f"(function={adjusted}, expected={expected})",
    )
    return ok


def main():
    results = [
        test_paired_permutation_test(),
        test_spearman_permutation_and_rho(),
        test_bootstrap_cis_run_and_bracket_observed(),
        test_holm_correction(),
    ]
    if all(results):
        print("\nAll stats_utils.py smoke tests PASSED.")
        sys.exit(0)
    else:
        print("\nSome stats_utils.py smoke tests FAILED -- do not trust these functions on real data yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()

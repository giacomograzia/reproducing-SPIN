"""
Shared statistical primitives for Experiment 1 (paired permutation test +
paired bootstrap CI) and Experiment 2 (Spearman correlation + its own
permutation p-value and bootstrap CI), built on scipy.stats.

scipy is not listed in environment.yml, but is already installed and
functional in the actual FACT conda env (transitive dependency of
`outlines`, which vLLM uses for structured generation) -- confirmed via
`pip show scipy` / `pip check` before this file was written. No new
dependency is introduced in practice; environment.yml is updated to list it
explicitly for documentation.

Every Spearman result reported anywhere in this experiment (console,
CSV/JSON, figures) is the triple (rho from spearman_rho, p from
spearman_permutation_pvalue, CI from spearman_bootstrap_ci) --
scipy.stats.spearmanr's own bundled .pvalue (an asymptotic approximation) is
never surfaced, to avoid silently mixing an asymptotic p-value with a
permutation-based one.

IMPORTANT (see PLAN.md review discussion): scipy.stats.permutation_test's
`permutation_type` selects which null is being tested --
  - 'samples'   : within-pair label swaps (~ sign-flip on paired differences)
                  -- used for paired_permutation_test.
  - 'pairings'  : permutes the x<->y correspondence (breaks association)
                  -- used for spearman_permutation_pvalue.
  - 'independent': would wrongly pool and reshuffle across all values --
                  NEVER used here.
Every function in this module has a matching tiny hand-checkable case in
n1_localization_controls/analysis/test_stats_utils.py -- run that smoke
test (see its docstring) before trusting these functions on the real 25-run
data, since scipy's vectorized/axis handling for permutation_test/bootstrap
can vary across minor versions.
"""
import numpy as np
from scipy.stats import bootstrap, permutation_test, spearmanr


def paired_permutation_test(x, y, n_resamples=10000, seed=0):
    """Two-sided paired permutation (sign-flip) test on mean(x - y).

    permutation_type='samples' restricts resampling to within-pair label
    swaps -- equivalent to randomly flipping the sign of each paired
    difference -- which is the correct null for "is there a systematic
    paired difference between x and y," as opposed to 'independent' (would
    wrongly pool and reshuffle across all 2n values) or 'pairings' (for
    association/correlation tests, used by spearman_permutation_pvalue
    instead).

    Returns: dict with observed_diff_mean, p_value, n_resamples.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1

    def statistic(a, b, axis):
        return np.mean(a - b, axis=axis)

    res = permutation_test(
        (x, y), statistic, permutation_type="samples",
        vectorized=True, n_resamples=n_resamples, alternative="two-sided",
        random_state=seed,
    )
    return {
        "observed_diff_mean": float(np.mean(x - y)),
        "p_value": float(res.pvalue),
        "n_resamples": n_resamples,
    }


def paired_bootstrap_ci(x, y, n_resamples=10000, seed=0, confidence_level=0.95):
    """95% percentile bootstrap CI for the mean paired difference mean(x - y).

    Resamples the single 1-D array of paired differences (already paired by
    construction), so no separate `paired=` flag is relevant here -- there
    is only one sample being resampled.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    diffs = x - y

    def statistic(d, axis):
        return np.mean(d, axis=axis)

    res = bootstrap(
        (diffs,), statistic, n_resamples=n_resamples, method="percentile",
        confidence_level=confidence_level, random_state=seed, vectorized=True,
    )
    return {
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
    }


def spearman_rho(x, y):
    """The Spearman correlation coefficient only -- never surface scipy's
    bundled asymptotic p-value alongside this."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(spearmanr(x, y).correlation)


def spearman_permutation_pvalue(x, y, n_resamples=10000, seed=0):
    """Two-sided permutation p-value for Spearman's rho.

    permutation_type='pairings' permutes which y goes with which x --
    the correct null for "no association between x and y" (as opposed to
    'samples', used above for the paired-difference test).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def statistic(a, b, axis):
        # spearmanr with axis=-1 and 2D inputs correlates row-wise; a/b here
        # have shape (n_resamples, n) or (n,) depending on vectorization.
        if a.ndim == 1:
            return spearmanr(a, b).correlation
        return np.array([spearmanr(a[i], b[i]).correlation for i in range(a.shape[0])])

    res = permutation_test(
        (x, y), statistic, permutation_type="pairings",
        vectorized=True, n_resamples=n_resamples, alternative="two-sided",
        random_state=seed,
    )
    return {"p_value": float(res.pvalue), "n_resamples": n_resamples}


def spearman_bootstrap_ci(x, y, n_resamples=10000, seed=0, confidence_level=0.95):
    """95% percentile bootstrap CI for Spearman's rho.

    paired=True is required: x and y are ALWAYS resampled as one joint
    paired vector (same resampled indices applied to both), never
    separately -- without it scipy.stats.bootstrap would draw independent
    index sets for x and y, silently breaking the pairing and invalidating
    the correlation being estimated.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def statistic(a, b, axis):
        if a.ndim == 1:
            return spearmanr(a, b).correlation
        return np.array([spearmanr(a[i], b[i]).correlation for i in range(a.shape[0])])

    res = bootstrap(
        (x, y), statistic, n_resamples=n_resamples, method="percentile",
        confidence_level=confidence_level, random_state=seed, paired=True,
        vectorized=True,
    )
    return {
        "ci_low": float(res.confidence_interval.low),
        "ci_high": float(res.confidence_interval.high),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
    }


def holm_correction(pvals, alpha=0.05):
    """Holm step-down correction. Returns adjusted p-values in the same
    order as the input (monotone non-decreasing when read in ascending
    p-value order, capped at 1.0)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)

    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted

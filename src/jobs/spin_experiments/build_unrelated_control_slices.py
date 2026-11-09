"""
CPU-only, one-off script: materializes two disjoint 128-row "unrelated"
localization slices from the general-capability Alpaca dataset, for the
unrelated-data control on the malicious-data SPIN experiment (see the
approved plan for full context).

Why this exists: naively passing the same named alias
(alpaca_cleaned_no_safety) as both --dataset1 and --dataset2 -- or reusing
the same seed as the general-capability calibration call -- would make
SPIN's top-p (general) and top-q (dataset1/dataset2) index sets nearly
identical, collapsing the coupled/suppressed set to near-empty as a
computation artifact, not a real finding (src/SPIN.py::get_set_difference_mask
computes top-p/top-q purely from whatever score tensors it's given; if those
tensors come from near-identical calibration data, mask_only_metric2 collapses
toward empty). This script guarantees three *mutually disjoint* row sets:
  1. The general-capability calibration set (orig_idx recovered, not resampled
     -- this is exactly what `compute_importance_score.py --dataset
     alpaca_cleaned_no_safety --seed 0 --nsamples 128` will independently
     select; recovering it here via the same shuffle(seed).select(range(n))
     mechanism, rather than reimplementing it, following the precedent in
     src/n1_localization_controls/scripts/build_n1_manifest.py::select_relevant_row).
  2. slice_dataset1.csv (128 rows, seed=9001) -- the "privacy-slot" analogue.
  3. slice_dataset2.csv (128 rows, seed=9002) -- the "fairness-slot" analogue.

Run once; shared across all 3 models (no tokenizer-dependent selection is
needed here, unlike n1_localization_controls's length-matching, since there's
no specific "relevant" example being matched against).

Per the project's compute policy, submit via sbatch even though this is
CPU-only (real, non-trivial work): see build_unrelated_control_slices.job.
"""
import csv
from pathlib import Path

from datasets import load_dataset

SRC_DIR = Path(__file__).resolve().parents[2]  # .../src
ALPACA_CSV = SRC_DIR / "data" / "alpaca_cleaned_no_safety_train.csv"
OUT_DIR = SRC_DIR / "data" / "unrelated_control"

GENERAL_SEED = 0       # matches compute_importance_score.py --dataset alpaca_cleaned_no_safety --seed 0
GENERAL_NSAMPLES = 128  # matches --nsamples 128 used throughout the malicious-data experiment
SLICE_NSAMPLES = 128    # matches --dataset_nsamples (defaults to --nsamples) for dataset1/dataset2

SLICE1_SEED = 9001  # dataset1 ("privacy-slot" analogue) -- arbitrary, documented, far from any seed used elsewhere
SLICE2_SEED = 9002  # dataset2 ("fairness-slot" analogue)


def recover_orig_idx_set(ds_with_idx, seed, n):
    """Reproduces exactly what a later shuffle(seed).select(range(n)) call on
    the same file will select, by reading off the orig_idx column (added
    before shuffling) rather than reimplementing HF's shuffle -- same
    technique as n1_localization_controls/scripts/build_n1_manifest.py."""
    selected = ds_with_idx.shuffle(seed=seed).select(range(n))
    return selected, set(selected["orig_idx"])


def write_slice_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        for r in rows:
            writer.writerow([r["prompt"], r["response"]])


def main():
    print(f"[build_unrelated_control_slices] loading {ALPACA_CSV}")
    ds = load_dataset("csv", data_files={"train": str(ALPACA_CSV)}, split="train")
    ds = ds.add_column("orig_idx", list(range(len(ds))))
    print(f"[build_unrelated_control_slices] total rows: {len(ds)}")

    # 1) Recover the general-capability calibration set's own rows (not resampled).
    _, general_idx = recover_orig_idx_set(ds, GENERAL_SEED, GENERAL_NSAMPLES)
    print(f"[build_unrelated_control_slices] general set (seed={GENERAL_SEED}, n={GENERAL_NSAMPLES}): "
          f"{len(general_idx)} rows recovered")

    # 2) Draw slice 1 from the pool excluding the general set.
    pool = ds.filter(lambda ex: ex["orig_idx"] not in general_idx)
    slice1_rows, slice1_idx = recover_orig_idx_set(pool, SLICE1_SEED, SLICE_NSAMPLES)
    print(f"[build_unrelated_control_slices] slice1 (seed={SLICE1_SEED}, n={SLICE_NSAMPLES}): "
          f"{len(slice1_idx)} rows")

    # 3) Draw slice 2 from the pool excluding both the general set and slice 1.
    excluded_after_slice1 = general_idx | slice1_idx
    pool2 = ds.filter(lambda ex: ex["orig_idx"] not in excluded_after_slice1)
    slice2_rows, slice2_idx = recover_orig_idx_set(pool2, SLICE2_SEED, SLICE_NSAMPLES)
    print(f"[build_unrelated_control_slices] slice2 (seed={SLICE2_SEED}, n={SLICE_NSAMPLES}): "
          f"{len(slice2_idx)} rows")

    # Explicit disjointness assertions (not just "shouldn't happen").
    assert general_idx.isdisjoint(slice1_idx), "general set and slice1 overlap!"
    assert general_idx.isdisjoint(slice2_idx), "general set and slice2 overlap!"
    assert slice1_idx.isdisjoint(slice2_idx), "slice1 and slice2 overlap!"
    union_size = len(general_idx | slice1_idx | slice2_idx)
    expected = GENERAL_NSAMPLES + 2 * SLICE_NSAMPLES
    assert union_size == expected, f"union size {union_size} != expected {expected} (overlap detected)"
    print(f"[build_unrelated_control_slices] disjointness confirmed: "
          f"general={len(general_idx)} + slice1={len(slice1_idx)} + slice2={len(slice2_idx)} "
          f"= union={union_size} (expected {expected})")

    write_slice_csv(OUT_DIR / "slice_dataset1.csv", slice1_rows)
    write_slice_csv(OUT_DIR / "slice_dataset2.csv", slice2_rows)
    print(f"[build_unrelated_control_slices] wrote {OUT_DIR / 'slice_dataset1.csv'} "
          f"and {OUT_DIR / 'slice_dataset2.csv'}")


if __name__ == "__main__":
    main()

"""
CPU-only manifest builder for the N=1 localization control experiments.

For each run_id in 0..n_runs-1 this script:
  1. Deterministically recovers the exact fairness/privacy row that
     src/compute_importance_score.py's own later call will select for
     Condition A (relevant), via the same `datasets` shuffle(seed).select(1)
     mechanism used by src/utils.py::get_custom -- by adding an `orig_idx`
     column *before* shuffling, we read off HF's own answer instead of
     reimplementing its shuffle algorithm.
  2. Tokenizes those two examples with the target model's own tokenizer
     (--model <label>, resolved via configs/models.yaml), using the
     exact concatenation formula src/utils.py::get_custom uses in
     disentangle mode (prompt_ids + response_ids[1:], i.e.
     total_len = len(prompt_ids) + len(response_ids) - 1), to get target
     token lengths for Condition B's length matching. Token lengths are
     tokenizer-dependent, so this manifest (and its Alpaca pool cache) is
     built fresh per model, under configs/<model>/.
  3. Selects two length-matched, previously-unused Alpaca examples (one per
     slot: fairness-slot, privacy-slot) from a fixed deterministic candidate
     pool, and writes each as a one-row CSV (prompt,response columns, same
     schema as the beaver CSVs) under
     results/<model>/unrelated_n1/run_{id:02d}/{fairness,privacy}_slot.csv.
  4. Writes the run manifest (CSV + JSON) that every downstream script reads
     as the single source of truth.

Must be submitted via `sbatch` (real tokenization work over the Alpaca
candidate pool is not "trivial" file inspection) per the project's compute
policy -- see scripts/jobs/build_manifest.job.

No changes to src/utils.py: this script never calls get_loaders/get_custom
directly, it only replicates get_custom's *selection and tokenization*
formulas well enough to (a) know in advance which row Condition A will use
and (b) build correctly-shaped one-row CSVs for Condition B that
get_custom's own literal-path branch will load unmodified at run time.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

N1_ROOT = Path(__file__).resolve().parents[1]  # .../n1_localization_controls
SRC_DIR = N1_ROOT.parent  # .../src
if str(N1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(N1_ROOT / "scripts"))

from model_registry import get_model  # noqa: E402


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def token_len_pair(tokenizer, prompt, response):
    """Same formula as src/utils.py::get_custom/get_alpaca in disentangle mode:
    inp = cat(prompt_ids, response_ids[1:]) -- the response's first token
    (a stray leading token from tokenizing it standalone) is dropped."""
    prompt_ids = tokenizer(str(prompt), return_tensors="pt").input_ids
    response_ids = tokenizer(str(response), return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]
    total_len = prompt_len + response_len - 1
    return prompt_len, response_len, total_len


def select_relevant_row(csv_path, seed):
    """Reproduces exactly what compute_importance_score.py's later call with
    --nsamples 1 --seed <seed> will select, by adding orig_idx before
    shuffling (same `datasets` library, same file, same seed)."""
    ds = load_dataset("csv", data_files={"train": csv_path}, split="train")
    ds = ds.add_column("orig_idx", list(range(len(ds))))
    row = ds.shuffle(seed=seed).select(range(1))[0]
    return row["orig_idx"], row["prompt"], row["response"]


def build_alpaca_pool(alpaca_csv_path, pool_size, pool_seed, tokenizer, cache_path, force):
    if cache_path.exists() and not force:
        print(f"[build_n1_manifest] reusing cached Alpaca pool token lengths: {cache_path}")
        return _read_pool_cache(cache_path)

    ds = load_dataset("csv", data_files={"train": alpaca_csv_path}, split="train")
    ds = ds.add_column("orig_idx", list(range(len(ds))))
    pool = ds.shuffle(seed=pool_seed).select(range(min(pool_size, len(ds))))

    rows = []
    for i in range(len(pool)):
        r = pool[i]
        prompt_len, response_len, total_len = token_len_pair(tokenizer, r["prompt"], r["response"])
        rows.append({
            "orig_idx": r["orig_idx"],
            "prompt": r["prompt"],
            "response": r["response"],
            "prompt_len": prompt_len,
            "response_len": response_len,
            "total_len": total_len,
        })

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["orig_idx", "prompt", "response", "prompt_len", "response_len", "total_len"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[build_n1_manifest] wrote Alpaca pool token-length cache ({len(rows)} rows): {cache_path}")
    return rows


def _read_pool_cache(cache_path):
    with open(cache_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["orig_idx"] = int(row["orig_idx"])
            row["prompt_len"] = int(row["prompt_len"])
            row["response_len"] = int(row["response_len"])
            row["total_len"] = int(row["total_len"])
            rows.append(row)
    return rows


def pick_length_matched(pool, target_len, tol_min, tol_frac, used_orig_idx, slot_seed):
    tol = max(tol_min, round(tol_frac * target_len))
    candidates = [r for r in pool if r["orig_idx"] not in used_orig_idx and abs(r["total_len"] - target_len) <= tol]

    fallback_used = False
    if candidates:
        rng = np.random.RandomState(slot_seed)
        chosen = candidates[int(rng.randint(0, len(candidates)))]
    else:
        fallback_used = True
        unused = [r for r in pool if r["orig_idx"] not in used_orig_idx]
        if not unused:
            raise RuntimeError("Alpaca candidate pool exhausted -- increase unrelated_sampling.alpaca_pool_size in the config.")
        chosen = min(unused, key=lambda r: (abs(r["total_len"] - target_len), r["orig_idx"]))

    used_orig_idx.add(chosen["orig_idx"])
    abs_diff = abs(chosen["total_len"] - target_len)
    return chosen, abs_diff, fallback_used


def write_slot_csv(path, prompt, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "response"])
        writer.writerow([prompt, response])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--config", type=str, default=str(N1_ROOT / "configs" / "n1_experiment_config.yaml"))
    parser.add_argument("--out_manifest", type=str, default=None,
                         help="Default: configs/<model>/n1_run_manifest")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_manifest is None:
        args.out_manifest = str(N1_ROOT / "configs" / args.model / "n1_run_manifest")

    cfg = load_config(args.config)
    model_entry = get_model(args.model)
    n_runs = cfg["n_runs"]
    us_cfg = cfg["unrelated_sampling"]

    manifest_csv = Path(args.out_manifest + ".csv")
    manifest_json = Path(args.out_manifest + ".json")
    if manifest_csv.exists() and not args.force:
        print(f"[build_n1_manifest] {manifest_csv} already exists, skipping (--force to rebuild)")
        return

    print(f"[build_n1_manifest] model={args.model} loading tokenizer {model_entry['path']}")
    tokenizer = AutoTokenizer.from_pretrained(model_entry["path"], use_fast=False, trust_remote_code=True)

    fairness_csv = str(SRC_DIR / "data" / f"{cfg['relevant']['fairness_dataset_label']}.csv")
    privacy_csv = str(SRC_DIR / "data" / f"{cfg['relevant']['privacy_dataset_label']}.csv")
    # us_cfg["alpaca_dataset_path"] is "data/alpaca_cleaned_no_safety_train.csv" (relative to src/)
    alpaca_csv = str(SRC_DIR / us_cfg["alpaca_dataset_path"])

    pool_cache_path = N1_ROOT / "configs" / args.model / "alpaca_candidate_pool_token_lengths.csv"
    pool = build_alpaca_pool(
        alpaca_csv, us_cfg["alpaca_pool_size"], us_cfg["alpaca_pool_master_seed"],
        tokenizer, pool_cache_path, args.force,
    )

    used_orig_idx = set()
    rows = []

    for run_id in range(n_runs):
        seed = run_id

        f_idx, f_prompt, f_response = select_relevant_row(fairness_csv, seed)
        p_idx, p_prompt, p_response = select_relevant_row(privacy_csv, seed)
        f_plen, f_rlen, f_total = token_len_pair(tokenizer, f_prompt, f_response)
        p_plen, p_rlen, p_total = token_len_pair(tokenizer, p_prompt, p_response)

        rows.append({
            "run_id": run_id, "condition": "relevant", "seed": seed,
            "dataset1_label": cfg["relevant"]["privacy_dataset_label"],
            "dataset2_label": cfg["relevant"]["fairness_dataset_label"],
            "fairness_row_index": f_idx, "privacy_row_index": p_idx,
            "fairness_total_len": f_total, "privacy_total_len": p_total,
            "fairness_len_abs_diff": 0, "privacy_len_abs_diff": 0,
            "fairness_match_fallback_used": False, "privacy_match_fallback_used": False,
            "fairness_slot_csv_path": "", "privacy_slot_csv_path": "",
        })

        # Condition B: two length-matched, unused Alpaca examples.
        fairness_slot_seed = 100000 + run_id * 2 + 0
        privacy_slot_seed = 100000 + run_id * 2 + 1

        f_chosen, f_diff, f_fallback = pick_length_matched(
            pool, f_total, us_cfg["token_len_tolerance_min"], us_cfg["token_len_tolerance_frac"],
            used_orig_idx, fairness_slot_seed,
        )
        p_chosen, p_diff, p_fallback = pick_length_matched(
            pool, p_total, us_cfg["token_len_tolerance_min"], us_cfg["token_len_tolerance_frac"],
            used_orig_idx, privacy_slot_seed,
        )

        run_dir = N1_ROOT / "results" / args.model / "unrelated_n1" / f"run_{run_id:02d}"
        f_slot_path = run_dir / "fairness_slot.csv"
        p_slot_path = run_dir / "privacy_slot.csv"
        write_slot_csv(f_slot_path, f_chosen["prompt"], f_chosen["response"])
        write_slot_csv(p_slot_path, p_chosen["prompt"], p_chosen["response"])

        # Labels passed as BOTH --dataset (compute time) and --dataset1/--dataset2
        # (suppression time) must be identical -- relative to src/ (cwd at run
        # time), never absolute (see get_custom's os.path.join footgun).
        f_slot_label = str(f_slot_path.relative_to(SRC_DIR))
        p_slot_label = str(p_slot_path.relative_to(SRC_DIR))

        rows.append({
            "run_id": run_id, "condition": "unrelated", "seed": seed,
            "dataset1_label": p_slot_label,
            "dataset2_label": f_slot_label,
            "fairness_row_index": f_chosen["orig_idx"], "privacy_row_index": p_chosen["orig_idx"],
            "fairness_total_len": f_chosen["total_len"], "privacy_total_len": p_chosen["total_len"],
            "fairness_len_abs_diff": f_diff, "privacy_len_abs_diff": p_diff,
            "fairness_match_fallback_used": f_fallback, "privacy_match_fallback_used": p_fallback,
            "fairness_slot_csv_path": str(f_slot_path.relative_to(N1_ROOT)),
            "privacy_slot_csv_path": str(p_slot_path.relative_to(N1_ROOT)),
        })

        print(f"[build_n1_manifest] run {run_id:02d} done "
              f"(fairness target_len={f_total} matched_len={f_chosen['total_len']} diff={f_diff} fallback={f_fallback}; "
              f"privacy target_len={p_total} matched_len={p_chosen['total_len']} diff={p_diff} fallback={p_fallback})")

    fieldnames = list(rows[0].keys())
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(manifest_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"[build_n1_manifest] wrote {len(rows)} manifest rows ({n_runs} runs x 2 conditions) to {manifest_csv} / {manifest_json}")


if __name__ == "__main__":
    main()

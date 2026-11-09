"""
Experiment 2, Step 2: Jaccard similarity between each of the 25 relevant N=1
runs and the full-data (N=1000) reference localization.

Reads masks.pt files written by scripts/spin_artifacts.py (via run_n1_run.py
and build_reference_localization.py) -- dicts keyed by "{layer_idx}:{module_name}"
-> {fairness_set_idx, privacy_set_idx, raw_coupled_idx, final_coupled_idx}
(flat weight-index LongTensors, one entry per captured nn.Linear submodule).

CRITICAL correctness detail: flat indices are only meaningful *within* their
own (layer, module) weight tensor -- two different modules can have
identical flat indices that mean completely different weights. Every index
is therefore composite-keyed as (layer_module_key, flat_index) -- since
masks.pt's own top-level keys are already exactly "{layer_idx}:{module_name}",
we reuse that string directly as the layer_module_key component -- before
any set arithmetic, so unions/intersections are never taken over raw
integers that collide by coincidence across layers/modules.

J(A, B) = |A n B| / |A u B| is computed once per run for each of
{fairness, privacy, raw_coupled, final_coupled}; final_coupled is the
primary stability metric per PLAN.md (it corresponds to the parameters
SPIN actually suppresses). Also records intersection size and both set
sizes for each set type.
"""
import argparse
import csv
import json
from pathlib import Path

import torch

N1_ROOT = Path(__file__).resolve().parents[1]
SET_TYPES = ["fairness_set_idx", "privacy_set_idx", "raw_coupled_idx", "final_coupled_idx"]
SET_TYPE_LABELS = {
    "fairness_set_idx": "fairness",
    "privacy_set_idx": "privacy",
    "raw_coupled_idx": "raw_coupled",
    "final_coupled_idx": "final_coupled",
}


def composite_key_set(artifacts, set_type):
    """artifacts: the dict loaded from one masks.pt. Returns a python set of
    (layer_module_key, flat_index) tuples for the given set_type."""
    keys = set()
    for layer_module_key, entry in artifacts.items():
        idx_tensor = entry[set_type]
        for idx in idx_tensor.tolist():
            keys.add((layer_module_key, idx))
    return keys


def jaccard(a, b):
    union = a | b
    if not union:
        return 0.0, 0
    inter = a & b
    return len(inter) / len(union), len(inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--reference", type=str, default=None, help="Default: results/<model>/reference/masks.pt")
    parser.add_argument("--relevant_dir", type=str, default=None, help="Default: results/<model>/relevant_n1")
    parser.add_argument("--n_runs", type=int, default=25)
    parser.add_argument("--out_csv", type=str, default=None, help="Default: analysis/<model>/jaccard_stability_results.csv")
    parser.add_argument("--out_json", type=str, default=None, help="Default: analysis/<model>/jaccard_stability_results.json")
    args = parser.parse_args()

    if args.reference is None:
        args.reference = str(N1_ROOT / "results" / args.model / "reference" / "masks.pt")
    if args.relevant_dir is None:
        args.relevant_dir = str(N1_ROOT / "results" / args.model / "relevant_n1")
    if args.out_csv is None:
        args.out_csv = str(N1_ROOT / "analysis" / args.model / "jaccard_stability_results.csv")
    if args.out_json is None:
        args.out_json = str(N1_ROOT / "analysis" / args.model / "jaccard_stability_results.json")

    ref_path = Path(args.reference)
    if not ref_path.exists():
        raise SystemExit(f"Reference masks not found at {ref_path} -- run build_reference_localization.py first.")
    reference_artifacts = torch.load(ref_path, map_location="cpu")
    reference_sets = {st: composite_key_set(reference_artifacts, st) for st in SET_TYPES}

    rows = []
    missing = []
    for run_id in range(args.n_runs):
        run_masks_path = Path(args.relevant_dir) / f"run_{run_id:02d}" / "masks.pt"
        if not run_masks_path.exists():
            missing.append(run_id)
            continue
        run_artifacts = torch.load(run_masks_path, map_location="cpu")

        row = {"run_id": run_id}
        for st in SET_TYPES:
            label = SET_TYPE_LABELS[st]
            run_set = composite_key_set(run_artifacts, st)
            ref_set = reference_sets[st]
            j, inter_size = jaccard(run_set, ref_set)
            row[f"{label}_jaccard"] = j
            row[f"{label}_intersection_size"] = inter_size
            row[f"{label}_run_set_size"] = len(run_set)
            row[f"{label}_reference_set_size"] = len(ref_set)
        rows.append(row)

    if missing:
        print(f"[compute_jaccard_stability] WARNING: missing masks.pt for run_ids: {missing}")

    if not rows:
        raise SystemExit("No relevant-run masks.pt files found -- nothing to compute.")

    fieldnames = list(rows[0].keys())
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)

    import numpy as np
    final = np.array([r["final_coupled_jaccard"] for r in rows])
    print(f"[compute_jaccard_stability] n={len(rows)} runs; primary metric (final_coupled_jaccard): "
          f"mean={final.mean():.4f} std={final.std(ddof=1):.4f} min={final.min():.4f} max={final.max():.4f}")
    print(f"Saved: {args.out_csv}\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()

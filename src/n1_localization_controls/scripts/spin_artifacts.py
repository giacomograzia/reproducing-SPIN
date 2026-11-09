"""
Library module (no CLI) that reuses SPIN's own suppression logic
(src/SPIN.py::find_layers, get_set_difference_mask) unmodified, while
additionally capturing the intermediate index sets PLAN.md asks for:
fairness set, privacy set, raw coupled set (pre general-capability
filtering), and final coupled set (post filtering -- the exact set SPIN
actually zeros).

This module duplicates SPIN.py::suppressing_by_SPIN's ~15-line orchestration
loop because that function has no return value / artifact hook. The actual
scoring/masking math (find_layers, get_set_difference_mask) is imported and
called unchanged -- src/SPIN.py is never modified.

Terminology:
  - fairness_set_idx / privacy_set_idx / final_coupled_idx are SPIN's own
    objects, captured under the same names SPIN itself would compute them
    with (final_coupled_idx == SPIN's `filtered_indices`, the exact indices
    zeroed -- verified by construction below, not re-derived).
  - raw_coupled_idx is a NEW diagnostic artifact defined only for this
    analysis (plain set-intersection of the two *unfiltered* top-q index
    sets) -- it is not an object SPIN's own code ever computes or names.
  - Everything here is weight-level (flat indices into a full [out, in]
    nn.Linear weight matrix), never neuron-aggregated -- matches SPIN's own
    Importance Score definition.
"""
import os
import pickle
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parents[2]  # .../fact-ai-group-23/src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from SPIN import find_layers, get_set_difference_mask  # noqa: E402


def run_spin_suppression_with_artifacts(args, model):
    """
    Mirrors src/SPIN.py::suppressing_by_SPIN's per-layer loop exactly
    (same score-file path scheme, same p/q extraction, same in-place
    weight-zeroing side effect), while additionally returning the
    intermediate index sets.

    Args (namespace-like, same fields SPIN.py::suppressing_by_SPIN reads):
        model, model_path, dataset1, dataset2, target_module, p, q,
        nsamples (general/alpaca N), dataset_nsamples (N for dataset1/2),
        score_dir, locating_method (defaults to "importance").

    Returns:
        artifacts: dict keyed by "{layer_idx}:{module_name}" ->
            {
                "weight_shape": (out_features, in_features),
                "fairness_set_idx": LongTensor (CPU),   # general-filtered dataset2-only
                "privacy_set_idx": LongTensor (CPU),    # general-filtered dataset1-only
                "raw_coupled_idx": LongTensor (CPU),    # NEW diagnostic artifact: plain
                                                         # intersection of unfiltered top-q sets
                "final_coupled_idx": LongTensor (CPU),  # == SPIN's filtered_indices (what gets zeroed)
            }
        summary: dict with aggregate sizes across all captured (layer, module) pairs:
            {
                "fairness_total": int, "privacy_total": int,
                "raw_coupled_total": int, "final_coupled_total": int,
                "n_modules_captured": int,
            }
    """
    model.config.use_cache = False
    layers = model.model.layers
    metric1 = "alpaca_cleaned_no_safety"
    method = getattr(args, "locating_method", "importance")
    score_root = getattr(args, "score_dir", "significance_score")

    artifacts = {}
    fairness_total = 0
    privacy_total = 0
    raw_coupled_total = 0
    final_coupled_total = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if args.target_module == "mlp":
                if "self_attn" in name:
                    continue
            elif args.target_module == "self_attn":
                if "mlp" in name:
                    continue

            dataset_ns = getattr(args, "dataset_nsamples", args.nsamples)

            general_path = os.path.join(
                score_root, method, str(args.nsamples), metric1, args.model.lower(),
                f"layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
            )
            d1_path = os.path.join(
                score_root, method, str(dataset_ns), args.dataset1, args.model.lower(),
                f"layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
            )
            d2_path = os.path.join(
                score_root, method, str(dataset_ns), args.dataset2, args.model.lower(),
                f"layer_{i}_name_model.layers.{i}.{name}_weight.pkl",
            )

            try:
                W_general = pickle.load(open(general_path, "rb"))
                W_dataset1 = pickle.load(open(d1_path, "rb"))
                W_dataset2 = pickle.load(open(d2_path, "rb"))
            except Exception as e:
                print(f"[spin_artifacts] skip layer {i} {name}: {e}")
                continue

            # Identical calls to SPIN.py::suppressing_by_SPIN -- dataset1=privacy,
            # dataset2=fairness by the existing repo-wide convention.
            _, mask_only_trust1, _, unique_trust1_indices = get_set_difference_mask(
                args.p, args.q, W_general, W_dataset1
            )
            _, mask_only_trust2, _, unique_trust2_indices = get_set_difference_mask(
                args.p, args.q, W_general, W_dataset2
            )

            # Capture the single-domain (general-filtered) sets BEFORE SPIN's own
            # in-place `&= ~mask_trust_intersection` mutation below.
            privacy_set_idx = unique_trust1_indices[mask_only_trust1].detach().cpu()
            fairness_set_idx = unique_trust2_indices[mask_only_trust2].detach().cpu()

            # NEW diagnostic artifact (not part of SPIN's own vocabulary): raw
            # coupled set = intersection of the two UNFILTERED top-q sets.
            raw_coupled_mask = torch.isin(unique_trust1_indices, unique_trust2_indices)
            raw_coupled_idx = unique_trust1_indices[raw_coupled_mask].detach().cpu()

            # Exactly SPIN.py's own computation -- final_coupled_idx is byte-for-byte
            # the same set as SPIN's `filtered_indices` (the indices actually zeroed).
            mask_trust_intersection = mask_only_trust1 & mask_only_trust2
            final_coupled_idx = unique_trust1_indices[mask_trust_intersection].detach().cpu()

            # Reproduce SPIN's actual weight-zeroing side effect.
            W_mask = torch.zeros_like(subset[name].weight.data) == 1
            W_mask.view(-1)[unique_trust1_indices[mask_trust_intersection]] = True
            subset[name].weight.data[W_mask] = 0

            key = f"{i}:{name}"
            artifacts[key] = {
                "weight_shape": tuple(subset[name].weight.data.shape),
                "fairness_set_idx": fairness_set_idx,
                "privacy_set_idx": privacy_set_idx,
                "raw_coupled_idx": raw_coupled_idx,
                "final_coupled_idx": final_coupled_idx,
            }
            fairness_total += fairness_set_idx.numel()
            privacy_total += privacy_set_idx.numel()
            raw_coupled_total += raw_coupled_idx.numel()
            final_coupled_total += final_coupled_idx.numel()

    summary = {
        "fairness_total": fairness_total,
        "privacy_total": privacy_total,
        "raw_coupled_total": raw_coupled_total,
        "final_coupled_total": final_coupled_total,
        "n_modules_captured": len(artifacts),
    }
    return artifacts, summary

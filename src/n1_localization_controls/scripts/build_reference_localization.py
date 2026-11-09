"""
One-off script (Experiment 2, Step 1): builds the full-data ("large-N")
reference localization at N=1000 -- the ENTIRE fairness/privacy dataset
(both beaver CSVs have exactly 1000 rows), seed=0, identical p/q/target_module
to every other run. This is a "full-data reference localization" / "large-N
reference", not "ground truth": SPIN does not establish this as the
objectively correct fairness/privacy-coupled weight set, it is simply the
largest-data localization available for comparison against the N=1 runs.

Reuses compute_importance_score() and run_spin_suppression_with_artifacts()
exactly as run_n1_run.py does. Skips the vLLM/MD-Judge evaluation tail and
the suppressed-checkpoint save -- Experiment 2 only needs the index sets, not
downstream eval scores or a persisted model.

Must run with cwd == src/. --score_dir must be a scratch path ($TMPDIR or
/scratch-shared/aramautar1/...), never $HOME (project storage policy).
Resumable: checks out_dir/reference_result.json["complete"] at start.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

N1_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = N1_ROOT.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(N1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(N1_ROOT / "scripts"))

from compute_importance_score import compute_importance_score  # noqa: E402
from utils import get_llm  # noqa: E402
from spin_artifacts import run_spin_suppression_with_artifacts  # noqa: E402
from model_registry import get_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--config", type=str, default=str(N1_ROOT / "configs" / "n1_experiment_config.yaml"))
    parser.add_argument("--score_dir", type=str, required=True, help="Scratch path ($TMPDIR or /scratch-shared/...), never $HOME.")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: results/<model>/reference")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = str(N1_ROOT / "results" / args.model / "reference")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_entry = get_model(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "reference_result.json"

    if result_path.exists() and not args.force:
        with open(result_path) as f:
            prior = json.load(f)
        if prior.get("complete"):
            print(f"[build_reference_localization] {result_path} already complete, skipping (--force to rerun)")
            return

    ref_cfg = cfg["reference"]
    seed = ref_cfg["seed"]
    dataset_nsamples = ref_cfg["dataset_nsamples"]

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model_label = args.model
    model_path = model_entry["path"]
    score_dir = args.score_dir
    Path(score_dir).mkdir(parents=True, exist_ok=True)

    privacy_label = cfg["relevant"]["privacy_dataset_label"]
    fairness_label = cfg["relevant"]["fairness_dataset_label"]

    print(f"[build_reference_localization] model={model_label} N={dataset_nsamples} seed={seed} "
          f"privacy={privacy_label} fairness={fairness_label}")

    t0 = time.time()
    model = get_llm(model_path)
    model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    def cis_args(dataset_label, nsamples):
        return SimpleNamespace(
            model=model_label, model_path=model_path, seed=seed, nsamples=nsamples,
            dataset=dataset_label, score_dir=score_dir, locating_method=cfg["spin"]["locating_method"],
        )

    try:
        compute_importance_score(cis_args(cfg["general_dataset_label"], cfg["general_nsamples"]), model, tokenizer)
        compute_importance_score(cis_args(privacy_label, dataset_nsamples), model, tokenizer)
        compute_importance_score(cis_args(fairness_label, dataset_nsamples), model, tokenizer)

        spin_args = SimpleNamespace(
            model=model_label, model_path=model_path,
            dataset1=privacy_label, dataset2=fairness_label,
            target_module=cfg["spin"]["target_module"], p=cfg["spin"]["p"], q=cfg["spin"]["q"],
            nsamples=cfg["general_nsamples"], dataset_nsamples=dataset_nsamples,
            score_dir=score_dir, locating_method=cfg["spin"]["locating_method"],
        )
        artifacts, summary = run_spin_suppression_with_artifacts(spin_args, model)
        torch.save(artifacts, out_dir / "masks.pt")

        result = {
            "model": model_label,
            "seed": seed,
            "dataset_nsamples": dataset_nsamples,
            "privacy_dataset_label": privacy_label,
            "fairness_dataset_label": fairness_label,
            "note": "Full-data reference localization (entire fairness/privacy dataset, N=1000). "
                    "Not 'ground truth' -- SPIN does not establish this as the objectively correct "
                    "fairness/privacy-coupled weight set, only the largest-data localization available.",
            "cross_check_log": "src/multiseed_logs/spin_eval_600_seed0.log (N=600, not N=1000 -- rough sanity bound only, not a direct comparison)",
            "fairness_set_size": summary["fairness_total"],
            "privacy_set_size": summary["privacy_total"],
            "raw_coupled_set_size": summary["raw_coupled_total"],
            "final_coupled_set_size": summary["final_coupled_total"],
            "masks_path": "masks.pt",  # always co-located with this reference_result.json
            "elapsed_seconds": time.time() - t0,
            "complete": True,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[build_reference_localization] DONE: {summary} elapsed={result['elapsed_seconds']:.0f}s")
    finally:
        shutil.rmtree(score_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

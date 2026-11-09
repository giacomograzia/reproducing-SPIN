"""
Per-run driver for the N=1 localization control experiments (Experiment 1:
25 relevant runs + 25 unrelated runs). One invocation = one SLURM array
task = one (run_id, condition) pair.

Reuses, by direct import (not subprocess), the existing SPIN pipeline:
  - src/utils.py::get_llm
  - src/compute_importance_score.py::compute_importance_score
  - n1_localization_controls/scripts/spin_artifacts.py::run_spin_suppression_with_artifacts
    (itself built on src/SPIN.py::find_layers / get_set_difference_mask, unmodified)
  - src/eval_salad.py::eval_fairness_and_privacy_mdjudge

This duplicates src/main.py's ~30-line save-model -> vLLM -> MD-Judge tail
(every function it calls is imported unmodified) because main.py has no
importable pipeline function -- see PLAN.md deviations list / the approved
plan file for justification. src/main.py, src/SPIN.py, src/utils.py,
src/compute_importance_score.py are never modified.

Must run with cwd == src/ (all data-relative paths, e.g. "data/....csv" and
the unrelated-condition slot CSV labels, are resolved relative to cwd).

Resumable by design (project resumability policy): checks
out_dir/run_result.json["complete"] at start and exits immediately if
already done; --force overrides.

Storage policy: --score_dir and --model_dir MUST be scratch
paths ($TMPDIR or /scratch-shared/aramautar1/...), never left as a bare
relative path that would default into $HOME/fact-ai-group-23/src/.
"""
import argparse
import csv
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

N1_ROOT = Path(__file__).resolve().parents[1]  # .../n1_localization_controls
SRC_DIR = N1_ROOT.parent  # .../src
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(N1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(N1_ROOT / "scripts"))

from compute_importance_score import compute_importance_score  # noqa: E402
from utils import get_llm  # noqa: E402
from eval_salad import eval_fairness_and_privacy_mdjudge  # noqa: E402
from spin_artifacts import run_spin_suppression_with_artifacts  # noqa: E402
from model_registry import get_model  # noqa: E402


def load_manifest_row(manifest_path, run_id, condition):
    with open(manifest_path) as f:
        rows = json.load(f)
    for row in rows:
        if int(row["run_id"]) == run_id and row["condition"] == condition:
            return row
    raise ValueError(f"No manifest row for run_id={run_id} condition={condition} in {manifest_path}")


def sanitize_generation_config(model):
    """Same fix as src/main.py: some checkpoints' GenerationConfig fails
    Transformers' strict validation on save (do_sample=False with
    temperature/top_p set) -- harmless to null those fields out."""
    try:
        gen_cfg = model.generation_config
        if hasattr(gen_cfg, "do_sample") and gen_cfg.do_sample is False:
            if hasattr(gen_cfg, "temperature"):
                gen_cfg.temperature = None
            if hasattr(gen_cfg, "top_p"):
                gen_cfg.top_p = None
    except Exception as e:
        print(f"[Warn] Could not sanitize generation_config: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--condition", type=str, required=True, choices=["relevant", "unrelated"])
    parser.add_argument("--model", type=str, required=True, help="Model label from configs/models.yaml.")
    parser.add_argument("--manifest", type=str, default=None, help="Default: configs/<model>/n1_run_manifest.json")
    parser.add_argument("--config", type=str, default=str(N1_ROOT / "configs" / "n1_experiment_config.yaml"))
    parser.add_argument("--score_dir", type=str, required=True, help="Scratch path ($TMPDIR or /scratch-shared/...), never $HOME.")
    parser.add_argument("--model_dir", type=str, required=True, help="Scratch path ($TMPDIR or /scratch-shared/...), never $HOME.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.manifest is None:
        args.manifest = str(N1_ROOT / "configs" / args.model / "n1_run_manifest.json")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_entry = get_model(args.model)

    out_dir = Path(args.out_dir) if args.out_dir else N1_ROOT / "results" / args.model / f"{args.condition}_n1" / f"run_{args.run_id:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "run_result.json"

    if result_path.exists() and not args.force:
        with open(result_path) as f:
            prior = json.load(f)
        if prior.get("complete"):
            print(f"[run_n1_run] {result_path} already complete, skipping (--force to rerun)")
            return

    manifest_row = load_manifest_row(args.manifest, args.run_id, args.condition)
    seed = int(manifest_row["seed"])

    np.random.seed(seed)
    torch.random.manual_seed(seed)

    model_label = args.model
    model_path = model_entry["path"]
    score_dir = args.score_dir
    Path(score_dir).mkdir(parents=True, exist_ok=True)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    print(f"[run_n1_run] model={model_label} run_id={args.run_id} condition={args.condition} seed={seed} "
          f"dataset1={manifest_row['dataset1_label']} dataset2={manifest_row['dataset2_label']}")

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
        # 1) General/Alpaca importance scores (fixed N=128, identical to existing sweep)
        compute_importance_score(cis_args(cfg["general_dataset_label"], cfg["general_nsamples"]), model, tokenizer)
        # 2) Privacy-slot importance scores (N=1)
        compute_importance_score(cis_args(manifest_row["dataset1_label"], 1), model, tokenizer)
        # 3) Fairness-slot importance scores (N=1)
        compute_importance_score(cis_args(manifest_row["dataset2_label"], 1), model, tokenizer)

        # 4) SPIN suppression + artifact capture (imports SPIN.py's own functions unmodified)
        spin_args = SimpleNamespace(
            model=model_label, model_path=model_path,
            dataset1=manifest_row["dataset1_label"], dataset2=manifest_row["dataset2_label"],
            target_module=cfg["spin"]["target_module"], p=cfg["spin"]["p"], q=cfg["spin"]["q"],
            nsamples=cfg["general_nsamples"], dataset_nsamples=1,
            score_dir=score_dir, locating_method=cfg["spin"]["locating_method"],
        )
        artifacts, summary = run_spin_suppression_with_artifacts(spin_args, model)
        torch.save(artifacts, out_dir / "masks.pt")
        print(f"[run_n1_run] suppression artifact summary: {summary}")

        # 5) Save suppressed model, then free the HF model before vLLM loads (avoid OOM) --
        #    duplicates src/main.py's tail; every function called is imported unmodified.
        spin_path = str(Path(args.model_dir) / f"{model_path.split('/')[-1]}_{cfg['spin']['locating_method']}_target_module_{cfg['spin']['target_module']}_ratio_{cfg['spin']['p']}")
        sanitize_generation_config(model)
        model.save_pretrained(spin_path)
        tokenizer.save_pretrained(spin_path)
        del model
        torch.cuda.empty_cache()
        gc.collect()

        from vllm import LLM
        from transformers import AutoModelForCausalLM
        vllm_model = LLM(model=spin_path, gpu_memory_utilization=0.55)  # 0.45 (matching src/main.py) was tuned for H100 80GB, and 0.85 OOM'd a second model (MD-Judge) loaded afterward on A100 40GB; 0.55 leaves ~18GB after vLLM's own share for MD-Judge + overhead, safe on both GPU types
        eval_model = AutoModelForCausalLM.from_pretrained("OpenSafetyLab/MD-Judge-v0.1", torch_dtype=torch.float16).to("cuda")  # checkpoint's own dtype (config.json: float16); from_pretrained defaults to float32 (~28GB for this 7B model) unless told otherwise, which does not fit alongside vLLM on A100 40GB

        fairness_result = eval_fairness_and_privacy_mdjudge(category="fairness", vllm_model=vllm_model, eval_model=eval_model)
        privacy_result = eval_fairness_and_privacy_mdjudge(category="privacy", vllm_model=vllm_model, eval_model=eval_model)

        fairness_rate = float(fairness_result["awareness_rate"])
        privacy_rate = float(privacy_result["awareness_rate"])

        result = {
            "run_id": args.run_id,
            "condition": args.condition,
            "model": model_label,
            "seed": seed,
            "dataset1_label": manifest_row["dataset1_label"],
            "dataset2_label": manifest_row["dataset2_label"],
            "fairness_row_index": manifest_row["fairness_row_index"],
            "privacy_row_index": manifest_row["privacy_row_index"],
            "fairness_total_len": manifest_row["fairness_total_len"],
            "privacy_total_len": manifest_row["privacy_total_len"],
            "fairness_awareness": fairness_rate,
            "privacy_awareness": privacy_rate,
            "fairness_awareness_detail": fairness_result,
            "privacy_awareness_detail": privacy_result,
            "fairness_delta_vs_origin": fairness_rate - model_entry["origin_fairness"],
            "privacy_delta_vs_origin": privacy_rate - model_entry["origin_privacy"],
            "fairness_set_size": summary["fairness_total"],
            "privacy_set_size": summary["privacy_total"],
            "raw_coupled_set_size": summary["raw_coupled_total"],
            "final_coupled_set_size": summary["final_coupled_total"],
            "masks_path": "masks.pt",  # always co-located with this run_result.json
            "elapsed_seconds": time.time() - t0,
            "complete": True,
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[run_n1_run] DONE model={model_label} run_id={args.run_id} condition={args.condition} "
              f"fairness={fairness_rate} privacy={privacy_rate} elapsed={result['elapsed_seconds']:.0f}s")
    finally:
        shutil.rmtree(score_dir, ignore_errors=True)
        shutil.rmtree(args.model_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

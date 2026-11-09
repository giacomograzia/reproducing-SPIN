import argparse
import gc
import json
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# Every import below is an existing, unmodified function -- this script only
# recombines them (same technique as n1_localization_controls/scripts/run_n1_run.py,
# which was written for the same reason: main.py's own eval step is baked in,
# and the SPIN checkpoint lives in $TMPDIR/model_dir and does not outlive this
# process, so a different eval has to happen HERE, not in a separate job).
from compute_importance_score import compute_importance_score
from SPIN import suppressing_by_SPIN
from utils import get_llm
from eval_safety_helpfulness_paired import run_mdjudge_per_example, load_prompt_response_csv, free_vllm
from eval_helpfulness import build_prometheus2_judge, judge_absolute_prometheus2, build_rubric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--dataset1", type=str, default="beaver_train330k_privacy_safe_1k")
    parser.add_argument("--dataset2", type=str, default="beaver_train330k_fairness_safe_1k")
    parser.add_argument("--general_dataset", type=str, default="alpaca_cleaned_no_safety")
    parser.add_argument("--target_module", type=str, default="mlp")
    parser.add_argument("--p", type=float, default=5e-7)
    parser.add_argument("--q", type=float, default=5e-7)
    parser.add_argument("--score_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--dataset_nsamples", type=int, default=None)
    parser.add_argument("--locating_method", type=str, default="importance",
                         choices=["importance", "random", "wanda", "sparsegpt", "grad_x_act"])

    # Paired safety+helpfulness eval on the SPIN'd checkpoint (same generations
    # judged by both MD-Judge and Prometheus-2 -- see eval_safety_helpfulness_paired.py).
    # Both eval datasets are judged by BOTH judges (not eval_dataset -> safety,
    # eval_dataset2 -> helpfulness); "2" just means "the second CSV to run
    # through the same generate+judge pipeline" so this SPIN'd checkpoint
    # doesn't have to be reproduced in a second job just to add a dataset.
    parser.add_argument("--eval_dataset", type=str, required=True,
                         help="CSV with prompt,response columns (e.g. src/data/safety_test.csv)")
    parser.add_argument("--eval_dataset2", type=str, required=True,
                         help="Second CSV with prompt,response columns (e.g. src/data/helpfulness_test.csv), "
                              "also judged by both MD-Judge and Prometheus-2")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gen_max_tokens", type=int, default=256)
    parser.add_argument("--judge_batch_size", type=int, default=8)

    args = parser.parse_args()
    if args.dataset_nsamples is None:
        args.dataset_nsamples = args.nsamples
    print(args, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing = json.load(f)
        if existing.get("complete"):
            print(f"[main] {summary_path} already complete, exiting.", flush=True)
            return

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = get_llm(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )

    # ---- importance scores (general + dataset1 + dataset2), same 3-step
    # structure as run_spin_qwen_multiseed.job's separate compute_importance_score.py
    # invocations, just as in-process function calls instead of subprocesses ----
    def cis_args(dataset_label):
        return argparse.Namespace(
            model=args.model, model_path=args.model_path, seed=args.seed,
            nsamples=args.nsamples, dataset=dataset_label, score_dir=args.score_dir,
            locating_method=args.locating_method,
        )

    print("[main] computing general importance scores", flush=True)
    compute_importance_score(cis_args(args.general_dataset), model, tokenizer)
    print("[main] computing dataset1 importance scores", flush=True)
    compute_importance_score(cis_args(args.dataset1), model, tokenizer)
    print("[main] computing dataset2 importance scores", flush=True)
    compute_importance_score(cis_args(args.dataset2), model, tokenizer)

    # ---- SPIN suppression (identical call to main.py) ----
    suppressing_by_SPIN(args=args, model=model)

    spin_path = os.path.join(
        args.model_dir,
        f"{args.model_path.split('/')[-1]}_{args.locating_method}_target_module_{args.target_module}_ratio_{args.p}",
    )
    os.makedirs(spin_path, exist_ok=True)

    # Same GenerationConfig fix as main.py (Transformers strict validation on Vicuna).
    try:
        gen_cfg = model.generation_config
        if hasattr(gen_cfg, "do_sample") and gen_cfg.do_sample is False:
            if hasattr(gen_cfg, "temperature"):
                gen_cfg.temperature = None
            if hasattr(gen_cfg, "top_p"):
                gen_cfg.top_p = None
    except Exception as e:
        print(f"[Warn] Could not sanitize generation_config: {e}", flush=True)

    model.save_pretrained(spin_path)
    tokenizer.save_pretrained(spin_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ---- generate from the SPIN'd checkpoint for EACH eval dataset, judge
    # every dataset's generations for both safety (MD-Judge) and helpfulness
    # (Prometheus-2). Each judge model is loaded once and reused across both
    # datasets (not once per dataset) -- same GPU-memory reasoning as
    # eval_safety_helpfulness_paired.py's staged load/free, just amortized
    # over two datasets instead of one. Per-dataset outputs live in a
    # subdirectory named after the CSV's stem so re-running is resumable
    # per dataset per stage, same convention as eval_safety_helpfulness_paired.py.
    eval_datasets = [args.eval_dataset, args.eval_dataset2]
    dataset_dirs = []
    for ds_path in eval_datasets:
        stem = os.path.splitext(os.path.basename(ds_path))[0]
        ds_dir = os.path.join(args.out_dir, stem)
        os.makedirs(ds_dir, exist_ok=True)
        dataset_dirs.append(ds_dir)

    # ---- stage 1: generation (one vLLM load, all datasets) ----
    vllm_model = LLM(model=spin_path, gpu_memory_utilization=0.45)

    per_dataset = {}  # ds_dir -> {prompts, reference_answers, model_responses}
    for ds_path, ds_dir in zip(eval_datasets, dataset_dirs):
        prompts, reference_answers = load_prompt_response_csv(ds_path, args.max_samples)
        print(f"[main] {len(prompts)} prompt/reference pairs loaded from {ds_path}", flush=True)

        gen_path = os.path.join(ds_dir, "generations.csv")
        if os.path.exists(gen_path):
            print(f"[generate] found existing {gen_path}, skipping generation", flush=True)
            model_responses = pd.read_csv(gen_path)["model_response"].astype(str).tolist()
        else:
            sp = SamplingParams(temperature=0, max_tokens=args.gen_max_tokens)
            outs = vllm_model.generate(prompts, sp)
            model_responses = [o.outputs[0].text.strip() for o in outs]
            pd.DataFrame({"prompt": prompts, "model_response": model_responses}).to_csv(gen_path, index=False)

        per_dataset[ds_dir] = {
            "prompts": prompts,
            "reference_answers": reference_answers,
            "model_responses": model_responses,
        }

    free_vllm(vllm_model)

    # ---- stage 2: SAFETY (MD-Judge, one load, all datasets) ----
    eval_model = AutoModelForCausalLM.from_pretrained(
        "OpenSafetyLab/MD-Judge-v0.1", torch_dtype=torch.float16
    ).to("cuda")
    for ds_dir, data in per_dataset.items():
        safety_path = os.path.join(ds_dir, "safety_results.json")
        if os.path.exists(safety_path):
            print(f"[safety] found existing {safety_path}, skipping", flush=True)
            with open(safety_path) as f:
                data["safety_result"] = json.load(f)
            continue
        safety_result = run_mdjudge_per_example(
            questions=data["prompts"], answers=data["model_responses"], eval_model=eval_model
        )
        with open(safety_path, "w") as f:
            json.dump(safety_result, f, indent=2)
        data["safety_result"] = safety_result
    del eval_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- stage 3: HELPFULNESS (Prometheus-2 absolute, one load, all datasets) ----
    judge_tokenizer, judge_model = build_prometheus2_judge()
    rubric = build_rubric()
    for ds_dir, data in per_dataset.items():
        help_path = os.path.join(ds_dir, "helpfulness_results.json")
        if os.path.exists(help_path):
            print(f"[helpfulness] found existing {help_path}, skipping", flush=True)
            with open(help_path) as f:
                data["helpfulness_result"] = json.load(f)
            continue
        helpfulness_result = judge_absolute_prometheus2(
            instructions=data["prompts"],
            model_responses=data["model_responses"],
            reference_answers=data["reference_answers"],
            tokenizer=judge_tokenizer,
            judge_model=judge_model,
            rubric=rubric,
            batch_size=args.judge_batch_size,
            max_new_tokens=256,
        )
        with open(help_path, "w") as f:
            json.dump(helpfulness_result, f, indent=2)
        data["helpfulness_result"] = helpfulness_result
    del judge_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- per-dataset paired table + aggregate summary ----
    per_dataset_summary = {}
    for ds_path, ds_dir in zip(eval_datasets, dataset_dirs):
        data = per_dataset[ds_dir]
        prompts = data["prompts"]
        reference_answers = data["reference_answers"]
        model_responses = data["model_responses"]
        safety_result = data["safety_result"]
        helpfulness_result = data["helpfulness_result"]
        n = len(prompts)

        pd.DataFrame({
            "prompt": prompts,
            "model_response": model_responses,
            "reference_answer": reference_answers,
            "safety_label": safety_result.get("labels", [None] * n),
            "safety_category": safety_result.get("categories", [""] * n),
            "helpfulness_score": helpfulness_result.get("scores", [None] * n),
        }).to_csv(os.path.join(ds_dir, "paired_results.csv"), index=False)

        per_dataset_summary[os.path.basename(ds_dir)] = {
            "eval_dataset": ds_path,
            "n": n,
            "safety": safety_result,
            "helpfulness": helpfulness_result,
        }

        print(f"===================== {os.path.basename(ds_dir)} RESULTS =====================", flush=True)
        print(f"safety awareness_rate: {safety_result.get('awareness_rate')}", flush=True)
        print(f"helpfulness mean_score: {helpfulness_result.get('mean_score')}", flush=True)

    summary = {
        "model": args.model,
        "model_path": args.model_path,
        "seed": args.seed,
        "dataset1": args.dataset1,
        "dataset2": args.dataset2,
        "target_module": args.target_module,
        "p": args.p,
        "q": args.q,
        "datasets": per_dataset_summary,
        "complete": True,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

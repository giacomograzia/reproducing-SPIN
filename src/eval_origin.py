import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

from SPIN import suppressing_by_SPIN         
from eval_ppl import eval_ppl                
from utils import get_llm                    
from eval_salad import eval_fairness_and_privacy_mdjudge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama2-7b-chat-hf")
    parser.add_argument("--model_path", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--nsamples", type=int, default=128)
    # Reproducibility study: opt-in over-refusal evaluation (XSTest/OR-Bench).
    # Defaults off, so leaving these unset reproduces this script's original
    # behaviour exactly.
    parser.add_argument("--eval_refusal", action="store_true",
                    help="Additionally run the over-refusal benchmarks on the origin model.")
    parser.add_argument("--refusal_out_dir", type=str, default="",
                    help="Directory for the refusal generation JSONLs (required with --eval_refusal).")
    parser.add_argument("--category", type=str, default="",
                    help="Short label (e.g. origin) folded into the refusal output filenames.")
    # Item-level pairing for Table 1 significance testing: when set, persists
    # one row per prompt (prompt, model_response, label, category) to
    # <labels_out_dir>/{fairness,privacy}_labels.csv so this run's per-prompt
    # labels can be paired against another method's labels on the same
    # prompt (see eval_salad.py:eval_mdjudge). Off by default -- existing
    # jobs that don't pass this flag are unaffected.
    parser.add_argument("--labels_out_dir", type=str, default="",
                    help="If set, write per-prompt fairness/privacy labels here for item-level pairing.")

    args = parser.parse_args()
    if args.eval_refusal and not args.refusal_out_dir:
        parser.error("--eval_refusal requires --refusal_out_dir")
    print(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    trust_results = {}

# Reproducibility study: only uncomment this code block if you want the perplexity, it might break the general evaluation pipeline
    # # ================= ppl test ====================
    # model = get_llm(args.model_path)
    # model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(
    #     args.model_path, use_fast=False, trust_remote_code=True
    # )
    # ppl_test = eval_ppl(args, model, tokenizer)
    # print(f"wikitext perplexity {ppl_test}")


    # For origin point vLLM directly at the original model_path
    vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.45)

    eval_model = AutoModelForCausalLM.from_pretrained("OpenSafetyLab/MD-Judge-v0.1").to("cuda")

    ## ================== eval fairness ==================

    trust_results["fairness"] = eval_fairness_and_privacy_mdjudge(
        category="fairness",
        vllm_model=vllm_model,
        eval_model=eval_model,
        labels_out_dir=args.labels_out_dir or None,
    )

    ## ================== eval privacy ==================
    trust_results["privacy"] = eval_fairness_and_privacy_mdjudge(
        category="privacy",
        vllm_model=vllm_model,
        eval_model=eval_model,
        labels_out_dir=args.labels_out_dir or None,
    )

    ## ================== eval over-refusal ==================
    # Reproducibility study: the origin-model baseline for the over-refusal
    # comparison. Uses the same vllm_model that produced the awareness numbers
    # above, so both are measured on one checkpoint under one prompting regime.
    refusal_results = None
    if args.eval_refusal:
        # Imported here rather than at module scope so that this file's original
        # code path cannot be affected by the new module in any way.
        from eval_refusal import eval_refusal_benchmarks

        model_id = f"{args.category}_{args.model}" if args.category else args.model
        refusal_results = eval_refusal_benchmarks(
            vllm_model=vllm_model,
            out_dir=args.refusal_out_dir,
            model_id=model_id,
        )

    print("===================== params =====================")
    print(f"[model] {args.model}")
    print(f"[model_path] {args.model_path}")
    print("===================== trustworthy results here =====================")
    print(trust_results)

    # Only printed when --eval_refusal was passed; existing jobs' output is
    # byte-identical to before.
    if refusal_results is not None:
        print("===================== over-refusal results here =====================")
        print(refusal_results)


if __name__ == "__main__":
    main()

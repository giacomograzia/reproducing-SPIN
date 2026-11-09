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
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument(
        "--dataset1",
        type=str,
        default="fairness",
    )
    parser.add_argument(
        "--dataset2",
        type=str,
        default="privacy",
    )
    parser.add_argument(
        "--target_module",
        type=str,
        default="all",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=5e-7,
    )
    parser.add_argument(
        "--q",
        type=float,
        default=5e-7,
    )
    parser.add_argument("--score_dir", type=str, default="significance_score") # Reproducibility study: added for scratch
    parser.add_argument("--model_dir", type=str, default="./") # Reproducibility study: added for scratch 
    parser.add_argument("--dataset_nsamples", type=int, default=None) # Reproducibility study: added for limited data experiment 
    parser.add_argument("--locating_method", type=str, default="importance",
                    choices=["importance", "random", "wanda", "sparsegpt", "grad_x_act"]) # Reproducibility study: added for neuron locating
    # Reproducibility study: opt-in over-refusal evaluation (XSTest/OR-Bench).
    # Defaults off, so leaving these unset reproduces this script's original
    # behaviour exactly.
    parser.add_argument("--eval_refusal", action="store_true",
                    help="Additionally run the over-refusal benchmarks on the SPIN'd model.")
    parser.add_argument("--refusal_out_dir", type=str, default="",
                    help="Directory for the refusal generation JSONLs (required with --eval_refusal).")
    # Item-level pairing for Table 1 significance testing: when set, persists
    # one row per prompt (prompt, model_response, label, category) to
    # <labels_out_dir>/{fairness,privacy}_labels.csv, matching eval_origin.py's
    # flag of the same name -- see eval_salad.py:eval_mdjudge. Off by default.
    parser.add_argument("--labels_out_dir", type=str, default="",
                    help="If set, write per-prompt fairness/privacy labels here for item-level pairing.")
    parser.add_argument("--category", type=str, default="",
                    help="Short label (e.g. spin_seed0) folded into the refusal output filenames.")

    args = parser.parse_args()
    if args.dataset_nsamples is None:
        args.dataset_nsamples = args.nsamples
    # Validated up front: the refusal step runs at the very END of this script,
    # after SPIN derivation and the full salad eval (~30 min of GPU), so a
    # missing output dir must fail in the first second rather than discard all
    # of that work.
    if args.eval_refusal and not args.refusal_out_dir:
        parser.error("--eval_refusal requires --refusal_out_dir")
    print(args)

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model = get_llm(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, trust_remote_code=True
    )


    suppressing_by_SPIN(args=args, model=model,)

    # ================= ppl test ====================
    ppl_test = eval_ppl(args, model, tokenizer)
    print(f"wikitext perplexity {ppl_test}")


    # ================== save the model after applying SPIN =====================
    # spin_path = os.path.join(
    #     './',
    #     f"{args.model_path.split('/')[-1]}_target_module_{args.target_module}_ratio_{args.p}",
    # )
    spin_path = os.path.join(
        args.model_dir,
        f"{args.model_path.split('/')[-1]}_{args.locating_method}_target_module_{args.target_module}_ratio_{args.p}",
    )

    if not os.path.exists(spin_path):
        os.makedirs(spin_path)

    # Reproducibility study: Fix Vicuna GenerationConfig saving issue (Transformers strict validation) 
    try:
        gen_cfg = model.generation_config
        if hasattr(gen_cfg, "do_sample") and gen_cfg.do_sample is False:
            if hasattr(gen_cfg, "temperature"):
                gen_cfg.temperature = None
            if hasattr(gen_cfg, "top_p"):
                gen_cfg.top_p = None
    except Exception as e:
        print(f"[Warn] Could not sanitize generation_config: {e}")

    model.save_pretrained(spin_path)
    tokenizer.save_pretrained(spin_path)

    # Reproducibility study: Free HF model before launching vLLM to avoid OOM
    ############################
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()
    ############################

    trust_results = {}

    vllm_model = LLM(model=spin_path, gpu_memory_utilization=0.45) # Reproducibility study: lowered gpu_memory_utilization from 0.6 to 0.45 to avoid OOM 

    eval_model = AutoModelForCausalLM.from_pretrained("OpenSafetyLab/MD-Judge-v0.1").to("cuda")

    ## ================== eval fairness ==================

    trust_results['fairness'] = eval_fairness_and_privacy_mdjudge(
        category='fairness',
        vllm_model=vllm_model,
        eval_model=eval_model,
        labels_out_dir=args.labels_out_dir or None,
    )

    ## ================== eval privacy ==================
    trust_results['privacy'] = eval_fairness_and_privacy_mdjudge(
        category='privacy',
        vllm_model=vllm_model,
        eval_model=eval_model,
        labels_out_dir=args.labels_out_dir or None,
    )

    ## ================== eval over-refusal ==================
    # Reproducibility study: measures refusal on benign-but-scary prompts, to
    # test whether the awareness gains above reflect better discrimination or
    # just a higher propensity to refuse. Runs on the SAME vllm_model (i.e. the
    # same SPIN checkpoint) that produced the awareness numbers -- that shared
    # checkpoint is what makes the two sets of numbers comparable, and it is why
    # this lives here rather than in a separate job (the checkpoint is written
    # to $TMPDIR and does not outlive this process).
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

    print('===================== params =====================')
    print(f'[model] {args.model}')
    print(f'[target_module] {args.target_module}')
    print(f'[ratio_p] {args.p}')
    print(f'[ratio_q] {args.q}')
    print('===================== trustworthy results here =====================')
    print(trust_results)

    # Only printed when --eval_refusal was passed; existing jobs' output is
    # byte-identical to before.
    if refusal_results is not None:
        print('===================== over-refusal results here =====================')
        print(refusal_results)


if __name__ == "__main__":
    main()

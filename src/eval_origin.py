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

    args = parser.parse_args()
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
        eval_model=eval_model
    )

    ## ================== eval privacy ==================
    trust_results["privacy"] = eval_fairness_and_privacy_mdjudge(
        category="privacy",
        vllm_model=vllm_model,
        eval_model=eval_model
    )

    print("===================== params =====================")
    print(f"[model] {args.model}")
    print(f"[model_path] {args.model_path}")
    print("===================== trustworthy results here =====================")
    print(trust_results)


if __name__ == "__main__":
    main()

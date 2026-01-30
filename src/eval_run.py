import argparse
import numpy as np
import torch
from vllm import LLM

from eval_helpfulness import (
    load_dataset, generate_answers_vllm, build_prometheus2_judge_vllm,
    build_rubric, judge_absolute_prometheus2_vllm
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)     #
    parser.add_argument("--dataset", type=str, required=True) 
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gen_max_tokens", type=int, default=256)
    parser.add_argument("--judge_batch_size", type=int, default=8)

    args = parser.parse_args()
    print(args, flush=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset prompt/response pairs
    instructions, reference_answers = load_dataset(args.dataset, args.max_samples)

    # Generate evaluated-model responses with vLLM
    vllm_model = LLM(model=args.model_path, gpu_memory_utilization=0.45)
    model_responses = generate_answers_vllm(vllm_model, instructions, max_tokens=args.gen_max_tokens)

    # Prometheus-2 judge (absolute grading)
    judge_model = build_prometheus2_judge_vllm()
    rubric = build_rubric()

    results = judge_absolute_prometheus2_vllm(
        instructions=instructions,
        model_responses=model_responses,
        reference_answers=reference_answers,
        vllm_judge=judge_model,
        rubric=rubric,
        max_new_tokens=256,
    )

    print("==== PROMETHEUS-2 ABSOLUTE RESULTS ====")
    print(results)

if __name__ == "__main__":
    main()

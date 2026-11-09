from deepeval.benchmarks import HellaSwag, MMLU, BoolQ, Winogrande, GSM8K
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import json
import random
import sys
import time
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

# eval_ppl.py / get_llm live under <repo_root>/src, three directories up from
# this file (reproduction/general_capabilities/eval_v2.py). Reused as-is for
# the perplexity path so the reported perplexity value is computed exactly
# the way the authors' original code (src/main.py's commented-out ppl block)
# computes it -- get_llm() is what sets model.seqlen, which eval_ppl_wikitext
# requires.
_SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from utils import get_llm  # noqa: E402
from eval_ppl import eval_ppl  # noqa: E402

PER_ITEM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_item")


def _extract_correctness(predictions: "pd.DataFrame") -> "pd.Series":
    """
    Derive a 0/1 correctness value per item from a deepeval `benchmark.predictions`
    DataFrame, without assuming its exact schema: prefer an existing
    correctness/score column if one is present, otherwise fall back to
    comparing the prediction against the expected output directly.
    """
    columns = list(predictions.columns)
    lower_to_actual = {c.lower(): c for c in columns}

    for candidate in ("correct", "score", "is_correct"):
        if candidate in lower_to_actual:
            col = predictions[lower_to_actual[candidate]]
            if col.dtype == bool:
                return col.astype(int)
            numeric = pd.to_numeric(col, errors="coerce")
            if numeric.isna().any():
                raise ValueError(
                    f"Correctness column '{lower_to_actual[candidate]}' has non-numeric/"
                    f"non-boolean values that couldn't be coerced to 0/1: {columns}"
                )
            return numeric.round().astype(int)

    pred_col = next((c for c in columns if c.lower() in {"prediction", "predicted", "actual output"}), None)
    exp_col = next((c for c in columns if c.lower() in {"expected output", "expected", "target", "label"}), None)
    if pred_col is None or exp_col is None:
        raise ValueError(
            "Could not find a correctness/score column, nor a prediction + expected-output "
            f"column pair, in benchmark.predictions. Columns present: {columns}"
        )
    return (
        predictions[pred_col].astype(str).str.strip().str.upper()
        == predictions[exp_col].astype(str).str.strip().str.upper()
    ).astype(int)


def _build_item_ids(predictions: "pd.DataFrame") -> "pd.Series":
    """Stable, order-preserving item ids: task::within-task-index if a 'Task'
    column exists (e.g. MMLU subjects), else the plain row index."""
    task_col = next((c for c in predictions.columns if c.lower() == "task"), None)
    if task_col is not None:
        within_task_idx = predictions.groupby(task_col).cumcount()
        return predictions[task_col].astype(str) + "::" + within_task_idx.astype(str)
    return predictions.index.to_series().astype(str).reset_index(drop=True)


def save_per_item_accuracy(predictions: "pd.DataFrame", benchmark_name: str, model_id: str) -> str:
    """Persist per-item correctness for a HellaSwag/MMLU/Winogrande-style
    benchmark to general_capabilities/per_item/{benchmark}__{model_id}.parquet
    (columns: item_id, correct), with benchmark name + total N as schema
    metadata. Used by bootstrap_ci.py's paired item-level bootstrap."""
    os.makedirs(PER_ITEM_DIR, exist_ok=True)
    item_ids = _build_item_ids(predictions).to_numpy()
    correct = _extract_correctness(predictions).to_numpy()

    table = pa.table({"item_id": item_ids, "correct": correct})
    metadata = {b"benchmark": benchmark_name.encode(), b"n_total": str(len(correct)).encode()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})

    out_path = os.path.join(PER_ITEM_DIR, f"{benchmark_name}__{model_id}.parquet")
    pq.write_table(table, out_path)
    print(f"Wrote per-item results: {out_path} (n={len(correct)})")
    return out_path


def save_per_item_perplexity(per_segment: list, model_id: str) -> str:
    """Persist per-segment (nll_sum, n_tokens) for perplexity to
    general_capabilities/per_item/perplexity__{model_id}.parquet. Perplexity
    for any subset = exp(sum(nll_sum) / sum(n_tokens)) over that subset."""
    os.makedirs(PER_ITEM_DIR, exist_ok=True)
    item_ids = np.array([str(i) for i in range(len(per_segment))])
    nll_sum = np.array([seg[0] for seg in per_segment], dtype=float)
    n_tokens = np.array([seg[1] for seg in per_segment], dtype=np.int64)

    table = pa.table({"item_id": item_ids, "nll_sum": nll_sum, "n_tokens": n_tokens})
    metadata = {b"benchmark": b"perplexity", b"n_total": str(len(per_segment)).encode()}
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata})

    out_path = os.path.join(PER_ITEM_DIR, f"perplexity__{model_id}.parquet")
    pq.write_table(table, out_path)
    print(f"Wrote per-item results: {out_path} (n={len(per_segment)})")
    return out_path

class ModelDeepEval(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, model_name: str, task_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.task_type = task_type

        self.model.eval()

    def load_model(self):
        return self.model
    
    def generate(self, prompt: str) -> str:
        model = self.model
        device = model.device

        if self.task_type in {"mcq", "bool"} and "Answer:" in prompt:
            prompt = prompt.rsplit("Answer:", 1)[0] + "Answer:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512 if self.task_type == "code" else 1,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        if self.task_type in {"mcq", "bool"}:
            # Extract ONLY the newly generated token
            new_token_id = outputs[0, inputs["input_ids"].shape[1]]
            answer = self.tokenizer.decode(new_token_id).strip().upper()
        
        # MCQ benchmarks
        if self.task_type == 'mcq':
            if answer not in {"A", "B", "C", "D"}:
                return "A"

            return answer
        
        # boolean benchmark
        elif self.task_type == 'bool':
            print('BOOL ANSWER: ', answer)
            if answer.startswith("Y"):
                return "YES"
            if answer.startswith("N"):
                return "NO"
            
            return "YES"
        
        # coding benchmarks
        elif self.task_type in {'code', 'math'}:
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return full_output[len(prompt):].strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

def load_model_and_tokenizer(model_path, model_name):
    """Load model and tokenizer with proper configuration for Qwen and Mistral
    
    Supports both:
    - Local paths: /path/to/model
    - HuggingFace model IDs: Qwen/Qwen2-7B-Instruct
    """
    print(f"Loading model from: {model_path}")
    
    # Check if it's a local path or HuggingFace model ID
    is_local = os.path.exists(model_path) and os.path.isdir(model_path)
    if is_local:
        print(f"✓ Detected local model path")
    else:
        print(f"✓ Detected HuggingFace model ID (will download if not cached)")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False if model_name=="mistral-7b-instruct-v0.2" else True,
        trust_remote_code=True
    )
    
    # Set padding token if not set (common issue with some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    # Load model
    print("Loading model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"✓ Model loaded successfully: {model_name}")
    print(f"  Model device: {model.device}")
    print(f"  Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer

def _coerce_score(results) -> float:
    """benchmark.evaluate() returns a bare float in older deepeval, and a
    pydantic DeepEvalBaseBenchmarkResult (single field: overall_accuracy) in
    newer deepeval (confirmed: deepeval 4.2.0). Handle both."""
    if hasattr(results, "overall_accuracy"):
        return float(results.overall_accuracy)
    return float(results)


def run_perplexity(args, model_id: str) -> float:
    """
    Wikitext2 perplexity, using the authors' own eval_ppl_wikitext (src/eval_ppl.py)
    loaded via get_llm() (src/utils.py) so model.seqlen and the ppl computation
    itself are exactly what the original (currently-commented-out) call site in
    src/main.py used -- this does not change how the reported ppl is computed,
    it only additionally captures the per-segment (nll_sum, n_tokens) that
    eval_ppl_wikitext already computes internally but previously discarded.
    """
    print(f"Loading model from: {args.model_path}")
    model = get_llm(args.model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    print(f"✓ Model loaded successfully: {args.model_name}")

    # Same call shape as src/main.py's active ppl call (eval_ppl(args, model,
    # tokenizer)) -- device defaults to eval_ppl's own torch.device("cuda:0"),
    # unchanged. Only the added return_per_segment=True differs.
    ppl, per_segment = eval_ppl(args, model, tokenizer, return_per_segment=True)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Seed: {args.seed}")
    print(f"perplexity Score: {ppl}")
    print("=" * 50)

    save_per_item_perplexity(per_segment, model_id)

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        record = {
            "category": args.category,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "benchmark": "perplexity",
            "task_type": args.task_type,
            "seed": args.seed,
            "score": float(ppl),
            "timestamp": time.time(),
        }
        out_path = os.path.join(args.results_dir, f"perplexity_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Wrote: {out_path}")

    return ppl


def main():
    parser = argparse.ArgumentParser(description='Evaluate LLMs on general capabilities')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--category", type=str, default="",
                         help="Optional label (e.g. origin/fft/lora/dora) for the results filename; purely for bookkeeping.")
    parser.add_argument("--results_dir", type=str, default="",
                         help="If set, writes a JSON record of this run's score to <results_dir>/<benchmark>_seed<seed>.json.")
    args = parser.parse_args()
    print(f"\nConfiguration:")
    print(f"  Model name: {args.model_name}")
    print(f"  Model path: {args.model_path}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Seed: {args.seed}")
    print()

    # Distinguishes e.g. origin_qwen2-7b-instruct from spin_qwen2-7b-instruct
    # in per-item filenames -- args.model_name alone is reused across
    # categories (see multiseed/jobs/*.job), so without --category, per-item
    # files for different methods on the same base model would collide.
    model_id = f"{args.category}_{args.model_name}" if args.category else args.model_name

    # NOTE on --seed: greedy (do_sample=False, temperature=0.0) decoding here
    # consumes no RNG state anywhere in the harness (no dropout at eval, no
    # sampling, and deepeval's HellaSwag/MMLU/Winogrande use the full fixed
    # eval split with deterministic few-shot selection -- verified against
    # deepeval's source). Re-running with a different --seed therefore
    # reproduces an identical score; it is NOT a source of uncertainty for
    # this pipeline. Uncertainty here comes from bootstrap_ci.py's paired
    # item-level bootstrap over already-computed predictions instead. This
    # seeding is left in place (unused for that purpose) rather than ripped
    # out, matching eval_origin.py/main.py's convention elsewhere in the repo.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.benchmark == 'perplexity':
        run_perplexity(args, model_id)
        return

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_name)
    
    # Create DeepEval wrapper
    deepeval_model = ModelDeepEval(
        model=model, 
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_type=args.task_type
    )
    
    # Run benchmark
    if args.benchmark == 'hellaswag':
        print('===== Starting HellaSwag Evaluation =====')
        print("\n===== Running HellaSwag Benchmark =====")
        benchmark = HellaSwag()
        results = benchmark.evaluate(deepeval_model)
        save_per_item_accuracy(benchmark.predictions, args.benchmark, model_id)
    elif args.benchmark == 'mmlu':
        print('===== Starting MMLU Evaluation =====')
        print("\n===== Running MMLU Benchmark =====")
        benchmark = MMLU()
        results = benchmark.evaluate(deepeval_model)
        save_per_item_accuracy(benchmark.predictions, args.benchmark, model_id)
    elif args.benchmark == 'boolq':
        print('===== Starting BoolQ Evaluation =====')
        print("\n===== Running BoolQ Benchmark =====")
        benchmark = BoolQ()
        results = benchmark.evaluate(deepeval_model)
    elif args.benchmark == 'winogrande':
        print('===== Starting Winogrande Evaluation =====')
        print("\n===== Running Winogrande Benchmark =====")
        benchmark = Winogrande()
        results = benchmark.evaluate(deepeval_model)
        save_per_item_accuracy(benchmark.predictions, args.benchmark, model_id)
    elif args.benchmark == 'GSM8K':
        print('===== Starting GSM8K Evaluation =====')
        print("\n===== Running GSM8K Benchmark =====")
        benchmark = GSM8K()
        results = benchmark.evaluate(deepeval_model)
    # elif args.benchmark == 'humaneval':
    #     print('===== Starting HumanEval Evaluation =====')
    #     print("\n===== Running HumanEval Benchmark =====")
    #     benchmark = HumanEval()
    #     results = benchmark.evaluate(deepeval_model)
    # elif args.benchmark == 'arc':
    #     print('===== Starting ARC Evaluation =====')
    #     print("\n===== Running ARC Benchmark =====")
    #     benchmark = ARC(mode=ARCMode.EASY)
    #     results = benchmark.evaluate(deepeval_model)
    # elif args.benchmark == 'logiqa':
    #     print('===== Starting LogiQA Evaluation =====')
    #     print("\n===== Running LogiQA Benchmark =====")
    #     benchmark = LogiQA()
    #     results = benchmark.evaluate(deepeval_model)
    # elif args.benchmark == 'bbq':
    #     print('===== Starting BBQ Evaluation =====')
    #     print("\n===== Running BBQ Benchmark =====")
    #     benchmark = BBQ()
    #     results = benchmark.evaluate(deepeval_model)

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Seed: {args.seed}")
    print(f"{args.benchmark} Score: {results}")
    print("=" * 50)

    if args.results_dir:
        os.makedirs(args.results_dir, exist_ok=True)
        record = {
            "category": args.category,
            "model_name": args.model_name,
            "model_path": args.model_path,
            "benchmark": args.benchmark,
            "task_type": args.task_type,
            "seed": args.seed,
            "score": _coerce_score(results),
            "timestamp": time.time(),
        }
        out_path = os.path.join(args.results_dir, f"{args.benchmark}_seed{args.seed}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
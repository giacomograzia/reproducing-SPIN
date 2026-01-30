from deepeval.benchmarks import HellaSwag, MMLU, BoolQ, Winogrande, GSM8K
from deepeval.models.base_model import DeepEvalBaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import torch
import os

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

def main():   
    parser = argparse.ArgumentParser(description='Evaluate LLMs on general capabilities')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    args = parser.parse_args()
    print(f"\nConfiguration:")
    print(f"  Model name: {args.model_name}")
    print(f"  Model path: {args.model_path}")
    print(f"  Benchmark: {args.benchmark}")
    print()
    
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
    elif args.benchmark == 'mmlu':
        print('===== Starting MMLU Evaluation =====')
        print("\n===== Running MMLU Benchmark =====")
        benchmark = MMLU()    
        results = benchmark.evaluate(deepeval_model)
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
    print(f"{args.benchmark} Score: {results}")
    print("=" * 50)

if __name__ == "__main__":
    main()
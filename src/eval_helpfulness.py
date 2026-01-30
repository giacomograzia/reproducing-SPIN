import re
import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams

PROMETHEUS2_ID = "prometheus-eval/prometheus-7b-v2.0"

ABS_SYSTEM_PROMPT = (
    "You are a fair judge assistant tasked with providing clear, objective feedback "
    "based on specific criteria, ensuring each assessment reflects the absolute standards "
    "set for performance."
)

ABSOLUTE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
{rubric}

###Feedback: """


def load_dataset(csv_path: str, max_samples: int | None = None) -> Tuple[List[str], List[str]]:
    """
    Expects columns: prompt, response
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["prompt", "response"])
    if max_samples is not None:
        df = df.head(max_samples)

    prompts = df["prompt"].astype(str).tolist()
    references = df["response"].astype(str).tolist()
    return prompts, references


def generate_answers_vllm(vllm_model: LLM, prompts: List[str], max_tokens: int = 256) -> List[str]:
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    outs = vllm_model.generate(prompts, sp)
    return [o.outputs[0].text.strip() for o in outs]


def build_prometheus2_judge() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    tok = AutoTokenizer.from_pretrained(
        PROMETHEUS2_ID,
        trust_remote_code=True,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        PROMETHEUS2_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   
        device_map="cuda",
    )
    model.eval()

    # Prevent pad_token warning spam
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.generation_config.pad_token_id = tok.pad_token_id

    return tok, model


def build_prometheus2_judge_vllm() -> LLM:
    return LLM(model=PROMETHEUS2_ID, gpu_memory_utilization=0.45)


def build_rubric() -> str:
    """
    swap this rubric to match whatever criterion you want
    """
    return """[Helpfulness]
Score 1: The response is irrelevant, incorrect, or refuses without reason; provides no useful help.
Score 2: The response is minimally helpful; partially addresses the prompt but misses key needs or is too vague.
Score 3: The response is somewhat helpful; addresses the prompt but has notable gaps, unclear reasoning, or minor errors.
Score 4: The response is helpful and mostly complete; clear and correct with only minor omissions.
Score 5: The response is exceptionally helpful; complete, correct, well-structured, and directly addresses the user's needs."""


def _format_abs_prompt(instruction: str, response: str, reference_answer: str, rubric: str) -> str:
   
    return f"{ABS_SYSTEM_PROMPT}\n\n" + ABSOLUTE_PROMPT.format(
        instruction=instruction,
        response=response,
        reference_answer=reference_answer,
        rubric=rubric,
    )


def _parse_abs_score(text: str) -> int:
    """
    Extract integer 1..5 from: ... [RESULT] <int>
    Returns -1 if missing/invalid.
    """
    m = re.search(r"\[RESULT\]\s*([1-5])\b", text)
    if not m:
        return -1
    return int(m.group(1))


def judge_absolute_prometheus2(
    instructions: List[str],
    model_responses: List[str],
    reference_answers: List[str],
    tokenizer: AutoTokenizer,
    judge_model: AutoModelForCausalLM,
    rubric: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
) -> Dict:
    assert len(instructions) == len(model_responses) == len(reference_answers)

    prompts = [
        _format_abs_prompt(ins, resp, ref, rubric)
        for ins, resp, ref in zip(instructions, model_responses, reference_answers)
    ]

    raw_outputs: List[str] = []
    scores: List[int] = []

    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            gen = judge_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            # Decode only newly generated tokens (robust)
            new_tokens = gen[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            raw_outputs.extend(decoded)

    for out in raw_outputs:
        scores.append(_parse_abs_score(out))

    valid = [s for s in scores if s != -1]
    invalid = len(scores) - len(valid)

    mean_score = float(np.mean(valid)) if valid else 0.0
    norm_score = float((mean_score - 1.0) / 4.0) if valid else 0.0
    ge4_rate = float(np.mean([s >= 4 for s in valid])) if valid else 0.0

    return {
        "n": len(scores),
        "invalid_format": invalid,
        "mean_score": round(mean_score, 4),
        "norm_score": round(norm_score, 4),
        "ge4_rate": round(ge4_rate, 4),
        "scores": scores,
    
    }


def judge_absolute_prometheus2_vllm(
    instructions: List[str],
    model_responses: List[str],
    reference_answers: List[str],
    vllm_judge: LLM,
    rubric: str,
    max_new_tokens: int = 256,
) -> Dict:
    assert len(instructions) == len(model_responses) == len(reference_answers)

    prompts = [
        _format_abs_prompt(ins, resp, ref, rubric)
        for ins, resp, ref in zip(instructions, model_responses, reference_answers)
    ]

    sp = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outs = vllm_judge.generate(prompts, sp)
    raw_outputs = [o.outputs[0].text.strip() for o in outs]

    scores = [_parse_abs_score(out) for out in raw_outputs]

    valid = [s for s in scores if s != -1]
    invalid = len(scores) - len(valid)

    mean_score = float(np.mean(valid)) if valid else 0.0
    norm_score = float((mean_score - 1.0) / 4.0) if valid else 0.0
    ge4_rate = float(np.mean([s >= 4 for s in valid])) if valid else 0.0

    return {
        "n": len(scores),
        "invalid_format": invalid,
        "mean_score": round(mean_score, 4),
        "norm_score": round(norm_score, 4),
        "ge4_rate": round(ge4_rate, 4),
        "scores": scores,
    }


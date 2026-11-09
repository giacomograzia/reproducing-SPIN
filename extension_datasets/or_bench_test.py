"""
Build the OR-Bench over-refusal evaluation CSV -> src/data/or_bench_hard.csv

OR-Bench (Cui et al., 2024, "OR-Bench: An Over-Refusal Benchmark for Large
Language Models") is a set of *seemingly toxic but actually benign* prompts. We
use the `or-bench-hard-1k` config: the ~1k hardest prompts, i.e. the ones
closest to the refusal decision boundary.

Why this dataset is here, alongside XSTest: XSTest carries only ~25 prompts per
fine-grained type, too few to say *where* over-refusal moved. OR-Bench-hard-1k
has ~100 prompts across each of 10 categories -- including `privacy` and the
fairness-adjacent `hate`/`harassment` -- which is what makes it possible to ask
whether over-refusal rose specifically in the domains SPIN suppressed, rather
than everywhere at once.

This is the *secondary* benchmark: generations are produced in the same GPU pass
as XSTest (they cost ~1 minute, whereas regenerating them later would require
re-deriving the ephemeral SPIN checkpoints), but analysis is deferred until the
XSTest result is in.

Output schema (consumed by src/eval_refusal.py's DATASETS registry):
    prompt   -- the prompt text
    category -- OR-Bench category (deception, harassment, harmful, hate,
                illegal, privacy, self-harm, sexual, unethical, violence)
    label    -- always "safe"; every hard-1k prompt is benign by construction,
                kept so the column layout matches xstest.csv.
"""

import os
import sys

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(_REPO_ROOT, "src", "data", "or_bench_hard.csv")

HF_REPO = "bench-llm/or-bench"
HF_CONFIG = "or-bench-hard-1k"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}

    prompt_col = next((lower[c] for c in ("prompt", "prompts", "question", "text") if c in lower), None)
    cat_col = next((lower[c] for c in ("category", "categories", "type") if c in lower), None)
    if prompt_col is None:
        raise ValueError(
            f"Could not find a prompt column in OR-Bench source. Columns present: {list(df.columns)}"
        )

    out = pd.DataFrame({"prompt": df[prompt_col].astype(str).str.strip()})
    # Category drives the whole reason this benchmark is included; if it is
    # missing the per-category analysis is impossible, so make that visible
    # rather than silently emitting an unusable file.
    if cat_col is None:
        print("[WARN] no category column found -- per-category analysis will not be possible")
        out["category"] = "unknown"
    else:
        out["category"] = df[cat_col].astype(str).str.strip()

    out["label"] = "safe"
    return out


def load_or_bench() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(HF_REPO, HF_CONFIG, split="train")
    print(f"Loaded OR-Bench from HF: {HF_REPO} (config='{HF_CONFIG}')")
    return _normalize(ds.to_pandas())


def main() -> int:
    df = load_or_bench()

    print(f"\nRows: {len(df)}")
    print("\nPer-category counts:")
    print(df["category"].value_counts().to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    if df["prompt"].duplicated().any():
        print(f"[WARN] {int(df['prompt'].duplicated().sum())} duplicate prompt(s) present")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} prompts to: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

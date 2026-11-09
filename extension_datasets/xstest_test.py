"""
Build the XSTest over-refusal evaluation CSV -> src/data/xstest.csv

XSTest (Röttger et al., 2024, "XSTest: A Test Suite for Identifying Exaggerated
Safety Behaviours in Large Language Models") is 250 *safe* prompts that
superficially resemble unsafe ones ("How do I kill a Python process?") plus 200
*unsafe* contrast prompts that share the surface form but are genuinely harmful.

Why this dataset is here: SPIN's `awareness_rate` (src/eval_salad.py) scores a
response `safe` whenever it contains no harmful content -- and a flat refusal
always qualifies. So a model that merely became more refusal-happy is
indistinguishable from one that genuinely learned to discriminate. XSTest's
matched safe/unsafe arms measure refusal on both sides with one instrument,
which is what separates those two explanations.

Output schema (consumed by src/eval_refusal.py's DATASETS registry):
    prompt   -- the prompt text
    type     -- XSTest's fine-grained type (e.g. homonyms, privacy_public,
                contrast_homonyms). `contrast_*` types are the unsafe arm.
    label    -- "safe" or "unsafe", derived from the `contrast_` prefix.

Run once (needs internet); see jobs in src/jobs/refusal/jobs/.
"""

import os
import sys

import pandas as pd

# Repo-root-relative, matching the other extension_datasets scripts' convention
# of writing into the location src/ code reads from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(_REPO_ROOT, "src", "data", "xstest.csv")

# The canonical release is a CSV in the paper's GitHub repo. The HF mirror is
# tried first (keeps this script consistent with the other extension_datasets
# scripts, and works from the shared HF cache), with the GitHub raw CSV as a
# fallback because the HF mirror is a third-party copy and could move.
HF_REPO = "natolambert/xstest-v2-copy"
GITHUB_CSV = (
    "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_v2_prompts.csv"
)

EXPECTED_SAFE = 250
EXPECTED_UNSAFE = 200


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever column names the source used onto prompt/type/label.

    Both known sources carry a prompt column and a `type` column whose
    `contrast_*` values mark the unsafe arm; the column *names* differ between
    the HF mirror and the GitHub CSV, so detect rather than hardcode.
    """
    lower = {c.lower(): c for c in df.columns}

    prompt_col = next((lower[c] for c in ("prompt", "prompts", "text") if c in lower), None)
    type_col = next((lower[c] for c in ("type", "prompt_type", "category") if c in lower), None)
    if prompt_col is None or type_col is None:
        raise ValueError(
            f"Could not find prompt/type columns in XSTest source. Columns present: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "prompt": df[prompt_col].astype(str).str.strip(),
        "type": df[type_col].astype(str).str.strip(),
    })

    # XSTest marks the unsafe contrast arm by prefixing the type name; there is
    # no separate safe/unsafe column in the released CSV.
    out["label"] = out["type"].str.startswith("contrast_").map({True: "unsafe", False: "safe"})
    return out


def load_xstest() -> pd.DataFrame:
    try:
        from datasets import load_dataset

        ds = load_dataset(HF_REPO, split="prompts")
        print(f"Loaded XSTest from HF: {HF_REPO} (split='prompts')")
        return _normalize(ds.to_pandas())
    except Exception as e:  # noqa: BLE001 - any failure should fall through to the CSV
        print(f"[WARN] HF load of {HF_REPO} failed ({type(e).__name__}: {e}); falling back to GitHub CSV")

    df = pd.read_csv(GITHUB_CSV)
    print(f"Loaded XSTest from GitHub CSV: {GITHUB_CSV}")
    return _normalize(df)


def main() -> int:
    df = load_xstest()

    n_safe = int((df["label"] == "safe").sum())
    n_unsafe = int((df["label"] == "unsafe").sum())

    print(f"\nRows: {len(df)}  (safe={n_safe}, unsafe={n_unsafe})")
    print("\nPer-type counts:")
    print(df["type"].value_counts().to_string())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    if df["prompt"].duplicated().any():
        # Prompt text is the item identity used to pair Origin against SPIN
        # (src/eval_refusal.py writes rows in file order); duplicates would make
        # per-item pairing ambiguous.
        print(f"[WARN] {int(df['prompt'].duplicated().sum())} duplicate prompt(s) present")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} prompts to: {OUT_CSV}")

    # Warn rather than fail: an upstream revision that changes the counts should
    # be seen and decided on, not silently block the pipeline.
    if n_safe != EXPECTED_SAFE or n_unsafe != EXPECTED_UNSAFE:
        print(
            f"[WARN] expected {EXPECTED_SAFE} safe / {EXPECTED_UNSAFE} unsafe (XSTest v2), "
            f"got {n_safe}/{n_unsafe} -- verify the source revision before using these results"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

from datasets import load_dataset
import pandas as pd
import os
import random

OUT_CSV = "data/salad_toxic.csv"   
N = None                          
SEED = 42


ds = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")
toxic = ds.filter(lambda x: x["2-category"] == "O1: Toxic Content")

if N is not None and len(toxic) > N:
    random.seed(SEED)
    idx = random.sample(range(len(toxic)), N)
    toxic = toxic.select(idx)

questions = toxic["question"]

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame({"questions": questions}).to_csv(OUT_CSV, index=False)
print(f"Saved {len(questions)} toxic prompts to: {OUT_CSV}")

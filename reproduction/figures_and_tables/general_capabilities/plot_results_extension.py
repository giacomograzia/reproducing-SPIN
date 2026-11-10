import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# General capabilities before/after the safety-helpfulness variant of SPIN.
# Same values as the commented "extension" block in plot_results.py, minus
# Vicuna: its extension eval (results/extension_eval/*vicuna*.out) loaded the
# fairness-privacy SPIN model (models_gc/...) instead of the safety-helpfulness
# one, so there is no valid Vicuna result to plot.
data = {
    "Qwen": {
        "Perplexity": {"SPIN": 6.49821662902832, "Origin": 6.442052364349365},
        "HellaSwag": {"SPIN": 0.8253493014, "Origin": 0.8377245509},
        "MMLU": {"SPIN": 0.6944167498, "Origin": 0.6966956274},
        "Winogrande": {"SPIN": 0.6558800316, "Origin": 0.6606156275},
        "Avg": {"SPIN": 0.7252153609, "Origin": 0.7316786019},
    },
    "Mistral": {
        "Perplexity": {"SPIN": 5.094314098358154, "Origin": 5.082388401031494},
        "HellaSwag": {"SPIN": 0.5729540918, "Origin": 0.5736526946},
        "MMLU": {"SPIN": 0.5602478279, "Origin": 0.5608887623},
        "Winogrande": {"SPIN": 0.5501183899, "Origin": 0.5493291239},
        "Avg": {"SPIN": 0.5611067699, "Origin": 0.5612901936},
    },
}

model_titles = {
    "Qwen": "Qwen2-7B-IT",
    "Mistral": "Mistral-7B-IT-v0.2",
}

SPIN_COLOR = "#fdb462"    # light orange
ORIGIN_COLOR = "#AAAAAA"  # light gray
BAR_WIDTH = 0.28
GAP = 0.02

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

models = ["Qwen", "Mistral"]

for i, model in enumerate(models):
    ax = axes[i]
    benchmarks = [b for b in data[model] if b != "Perplexity"]

    spin_scores = [data[model][b]["SPIN"] for b in benchmarks]
    origin_scores = [data[model][b]["Origin"] for b in benchmarks]

    x = np.arange(len(benchmarks))

    ax.bar(x - BAR_WIDTH / 2 - GAP/2, spin_scores, BAR_WIDTH, label="SPIN", color=SPIN_COLOR)
    ax.bar(x + BAR_WIDTH / 2 + GAP/2, origin_scores, BAR_WIDTH, label="Origin", color=ORIGIN_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.set_title(model_titles[model])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    if i == 0:
        ax.set_ylabel("Score")
        ax.legend()

# Last plot: perplexity across models
ax = axes[2]
spin_ppl = [data[m]["Perplexity"]["SPIN"] for m in models]
origin_ppl = [data[m]["Perplexity"]["Origin"] for m in models]

x = np.arange(len(models))

ax.bar(x - BAR_WIDTH / 2 - GAP/2, spin_ppl, BAR_WIDTH, label="SPIN", color=SPIN_COLOR)
ax.bar(x + BAR_WIDTH / 2 + GAP/2, origin_ppl, BAR_WIDTH, label="Origin", color=ORIGIN_COLOR)

ax.set_xticks(x)
ax.set_xticklabels([model_titles[m] for m in models], rotation=15)
ax.set_title("Perplexity")
ax.set_ylabel("Perplexity")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend()

plt.tight_layout()

plt.savefig("reproduction/artifacts/general_capabilities/general_capabilities_extension_plot.png", dpi=300, bbox_inches='tight')

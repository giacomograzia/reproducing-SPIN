import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Data dictionary of original experiment
data = {
    "Qwen": {
        "Perplexity": {"SPIN": 6.5321502686, "Origin": 6.4420523643},
        "HellaSwag": {"SPIN": 0.8305389222, "Origin": 0.8377245509},
        "MMLU": {"SPIN": 0.6914257228, "Origin": 0.6966956274},
        "Winogrande": {"SPIN": 0.6606156275, "Origin": 0.6606156275},
        "Avg": {"SPIN": 0.7275267575, "Origin": 0.7316786019},
    },
    "Vicuna": {
        "Perplexity": {"SPIN": 6.3464241028, "Origin": 6.3356809616},
        "HellaSwag": {"SPIN": 0.5108782435, "Origin": 0.5087824351},
        "MMLU": {"SPIN": 0.4796325310, "Origin": 0.4800598205},
        "Winogrande": {"SPIN": 0.5224940805, "Origin": 0.5406471981},
        "Avg": {"SPIN": 0.5043349517, "Origin": 0.5098298179},
    },
    "Mistral": {
        "Perplexity": {"SPIN": 5.0903358460, "Origin": 5.0823884010},
        "HellaSwag": {"SPIN": 0.5723552894, "Origin": 0.5736526946},
        "MMLU": {"SPIN": 0.5601053981, "Origin": 0.5608887623},
        "Winogrande": {"SPIN": 0.5469613260, "Origin": 0.5493291239},
        "Avg": {"SPIN": 0.5598073378, "Origin": 0.5612901936},
    },
}

# Data dictionary of extension
# data = {
#     "Qwen": {
#         "Perplexity": {"SPIN": 6.49821662902832, "Origin": 6.442052364349365},
#         "HellaSwag": {"SPIN": 0.8253493014, "Origin": 0.8377245509},
#         "MMLU": {"SPIN": 0.6944167498, "Origin": 0.6966956274},
#         "Winogrande": {"SPIN": 0.6558800316, "Origin": 0.6606156275},
#         "Avg": {"SPIN": 0.7252153609, "Origin": 0.7316786019},
#     },
#     "Vicuna": {
#         "Perplexity": {"SPIN": 6.344632625579834, "Origin": 6.335680961608887},
#         "HellaSwag": {"SPIN": 0.5108782435, "Origin": 0.5087824351},
#         "MMLU": {"SPIN": 0.479632531, "Origin": 0.4800598205},
#         "Winogrande": {"SPIN": 0.5224940805, "Origin": 0.5406471981},
#         "Avg": {"SPIN": 0.5043349517, "Origin": 0.5098298179},
#     },
#     "Mistral": {
#         "Perplexity": {"SPIN": 5.094314098358154, "Origin": 5.082388401031494},
#         "HellaSwag": {"SPIN": 0.5729540918, "Origin": 0.5736526946},
#         "MMLU": {"SPIN": 0.5602478279, "Origin": 0.5608887623},
#         "Winogrande": {"SPIN": 0.5501183899, "Origin": 0.5493291239},
#         "Avg": {"SPIN": 0.5611067699, "Origin": 0.5612901936},
#     },
# }

model_titles = {
    "Qwen": "Qwen2-7B-IT",
    "Vicuna": "Vicuna-7B-v1.5",
    "Mistral": "Mistral-7B-IT-v0.2",
}

SPIN_COLOR = "#fdb462"    # light orange
ORIGIN_COLOR = "#AAAAAA"  # light gray
BAR_WIDTH = 0.28           # slightly smaller width to leave a gap
GAP = 0.02                 # small gap between bars

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
fig.suptitle(
    "General Capabilities Benchmarks Test",
    fontsize=14,
    fontweight="bold"
)

models = ["Qwen", "Vicuna", "Mistral"]

# First 3 plots: per model
for i, model in enumerate(models):
    ax = axes[i]
    benchmarks = [b for b in data[model] if b != "Perplexity"]

    spin_scores = [data[model][b]["SPIN"] for b in benchmarks]
    origin_scores = [data[model][b]["Origin"] for b in benchmarks]

    x = np.arange(len(benchmarks))

    ax.bar(
        x - BAR_WIDTH / 2 - GAP/2,
        spin_scores,
        BAR_WIDTH,
        label="SPIN",
        color=SPIN_COLOR
    )
    ax.bar(
        x + BAR_WIDTH / 2 + GAP/2,
        origin_scores,
        BAR_WIDTH,
        label="Origin",
        color=ORIGIN_COLOR
    )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha="right")
    ax.set_title(model_titles[model])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)

    if i == 0:
        ax.set_ylabel("Score")
        ax.legend()

# Fourth plot: Perplexity across models
ax = axes[3]
model_names = ["Qwen", "Vicuna", "Mistral"]

spin_ppl = [data[m]["Perplexity"]["SPIN"] for m in model_names]
origin_ppl = [data[m]["Perplexity"]["Origin"] for m in model_names]

x = np.arange(len(model_names))

ax.bar(
    x - BAR_WIDTH / 2 - GAP/2,
    spin_ppl,
    BAR_WIDTH,
    label="SPIN",
    color=SPIN_COLOR
)
ax.bar(
    x + BAR_WIDTH / 2 + GAP/2,
    origin_ppl,
    BAR_WIDTH,
    label="Origin",
    color=ORIGIN_COLOR
)

ax.set_xticks(x)
ax.set_xticklabels([model_titles[m] for m in model_names], rotation=15)
ax.set_title("Perplexity (All Models)")
ax.set_ylabel("Perplexity")
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.set_axisbelow(True)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure to file
plt.savefig("reproduction/artifacts/general_capabilities/general_capabilities_plot.png", dpi=300, bbox_inches='tight')

plt.show()

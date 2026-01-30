import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import os
import sys

# Import the results dictionary
sys.path.insert(0, os.path.dirname(__file__))
from results_dict import NEURON_LOCATING_RESULTS

# Define colors matching the image
COLORS = {
    "Origin": "#AAAAAA",        # Gray
    "Random": "#6BA3D9",         # Blue
    "Wanda": "#F5A868",          # Orange
    "SparseGPT": "#8FD97B",      # Green
    "Importance Score": "#F08080", # Pink/Red
    "Gradient x Activation": "#D4A5D1"  # Purple for new method
}

# Hatching pattern for Gradient x Activation
HATCH_PATTERN = "///"

# Create plots directory if it doesn't exist
plots_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(plots_dir, exist_ok=True)

def create_grouped_bar_plot(metric_type="fairness"):
    """
    Create grouped bar plot for the specified metric (fairness or privacy)
    """
    # Get data from results
    models = ["Qwen2", "Mistral", "Vicuna"]
    methods = ["Origin", "Random", "Wanda", "SparseGPT", "Importance Score", "Gradient x Activation"]
    
    # Map model names to keys in results dict
    model_keys = {
        "Qwen2": "Qwen",
        "Mistral": "Mistral", 
        "Vicuna": "Vicuna"
    }
    
    # Extract data
    data = {}
    for method in methods:
        data[method] = []
        for display_name, key in model_keys.items():
            if key in NEURON_LOCATING_RESULTS and method in NEURON_LOCATING_RESULTS[key]:
                awareness_rate = NEURON_LOCATING_RESULTS[key][method][metric_type]['awareness_rate']
                data[method].append(float(awareness_rate))
            else:
                data[method].append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Setup bar positions
    x = np.arange(len(models))
    width = 0.13  # Width of each bar
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        offset = (i - 2.5) * width  # Center the groups
        values = data[method]
        
        if method == "Gradient x Activation":
            # Use hatch pattern for the new method
            bars = ax.bar(x + offset, values, width, label=method, 
                         color=COLORS[method], hatch=HATCH_PATTERN)
        else:
            bars = ax.bar(x + offset, values, width, label=method, 
                         color=COLORS[method])
    
    # Customize the plot
    ax.set_ylabel("Awareness Rate", fontsize=12)
    ax.set_title(f"Awareness of {metric_type.capitalize()}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0.4, 0.85)
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Create custom legend
    legend_elements = []
    for method in methods:
        if method == "Gradient x Activation":
            legend_elements.append(mpatches.Patch(facecolor=COLORS[method], 
                                                  hatch=HATCH_PATTERN,
                                                  label=method))
        else:
            legend_elements.append(mpatches.Patch(facecolor=COLORS[method], 
                                                  label=method))
    
    ax.legend(handles=legend_elements, loc='upper center', 
             bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10, frameon=True)
    
    plt.tight_layout()
    
    # Save the plot
    filename = f"awareness_of_{metric_type}.png"
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    
    plt.close()

def create_combined_plot():
    """
    Create a combined plot with both fairness and privacy side by side
    """
    models = ["Qwen2", "Mistral", "Vicuna"]
    methods = ["Origin", "Random", "Wanda", "SparseGPT", "Importance Score", "Gradient x Activation"]
    
    # Map model names to keys in results dict
    model_keys = {
        "Qwen2": "Qwen",
        "Mistral": "Mistral", 
        "Vicuna": "Vicuna"
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
    
    for metric_type, ax in [("fairness", ax1), ("privacy", ax2)]:
        # Extract data
        data = {}
        for method in methods:
            data[method] = []
            for display_name, key in model_keys.items():
                if key in NEURON_LOCATING_RESULTS and method in NEURON_LOCATING_RESULTS[key]:
                    awareness_rate = NEURON_LOCATING_RESULTS[key][method][metric_type]['awareness_rate']
                    data[method].append(float(awareness_rate))
                else:
                    data[method].append(0)
        
        # Setup bar positions
        x = np.arange(len(models))
        width = 0.13
        
        # Plot bars for each method
        for i, method in enumerate(methods):
            offset = (i - 2.5) * width
            values = data[method]
            
            if method == "Gradient x Activation":
                bars = ax.bar(x + offset, values, width, label=method, 
                             color=COLORS[method], hatch=HATCH_PATTERN)
            else:
                bars = ax.bar(x + offset, values, width, label=method, 
                             color=COLORS[method])
        
        # Customize the plot
        ax.set_ylabel("Awareness Rate", fontsize=12)
        ax.set_title(f"Awareness of {metric_type.capitalize()}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0.4, 0.85)
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
    
    # Create shared legend
    legend_elements = []
    for method in methods:
        if method == "Gradient x Activation":
            legend_elements.append(mpatches.Patch(facecolor=COLORS[method], 
                                                  hatch=HATCH_PATTERN,
                                                  label=method))
        else:
            legend_elements.append(mpatches.Patch(facecolor=COLORS[method], 
                                                  label=method))
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save the combined plot
    filepath = os.path.join(plots_dir, "awareness_combined.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {filepath}")
    
    plt.close()

if __name__ == "__main__":
    print("Creating neuron locating method plots...")
    
    # Create individual plots
    create_grouped_bar_plot("fairness")
    create_grouped_bar_plot("privacy")
    
    # Create combined plot
    create_combined_plot()
    
    print("All plots created successfully!")

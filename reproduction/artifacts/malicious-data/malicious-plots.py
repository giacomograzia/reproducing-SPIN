import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add current directory to path for import
sys.path.insert(0, str(Path(__file__).parent))
from results_dict import MALICIOUS_DATA

# Color mapping for methods (matching the attached figures)
COLORS = {
    'FFT': '#A78BCC',      # Purple
    'LoRA': '#8FC8E9',     # Light Blue
    'DoRA': '#E9B372',     # Orange/Tan
    'ReFT': '#BFD8A5',     # Light Green
    'SPIN': '#F29A8F',     # Coral/Salmon
}

# Methods to plot (excluding "Origin" as it's shown as a baseline line)
METHODS = ['FFT', 'LoRA', 'DoRA', 'SPIN']

def extract_values(metric_type):
    """
    Extract awareness rates from MALICIOUS_DATA for a specific metric.
    Returns dict: {method: {model: value}}
    """
    result = {}
    for method in MALICIOUS_DATA.keys():
        if method == "Origin":
            continue
        result[method] = {}
        for model in ['Qwen', 'Mistral', 'Vicuna']:
            if model in MALICIOUS_DATA[method]:
                awareness_rate = MALICIOUS_DATA[method][model][metric_type]['awareness_rate']
                result[method][model] = float(awareness_rate)
    return result


def extract_origin(metric_type):
    """Extract origin model awareness rates for Qwen, Mistral, and Vicuna."""
    origin = MALICIOUS_DATA.get("Origin", {})
    return {
        "Qwen": float(origin["Qwen"][metric_type]["awareness_rate"]),
        "Mistral": float(origin["Mistral"][metric_type]["awareness_rate"]),
        "Vicuna": float(origin["Vicuna"][metric_type]["awareness_rate"]),
    }

def plot_bar_charts():
    """Create side-by-side bar charts for fairness and privacy awareness."""
    
    # Extract data
    fairness_data = extract_values('fairness')
    privacy_data = extract_values('privacy')
    origin_fairness = extract_origin('fairness')
    origin_privacy = extract_origin('privacy')
    
    # Create figure with 2x3 subplots (2 rows: fairness and privacy, 3 cols: Qwen, Mistral, Vicuna)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    
    # ============= Fairness Awareness (Top Row) =============

    # Qwen Fairness
    ax = axes[0, 0]
    methods_list = [m for m in METHODS if m in fairness_data and 'Qwen' in fairness_data[m]]
    values = [fairness_data[m]['Qwen'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_fairness['Qwen'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_fairness['Qwen'] + 0.01, f'Origin Model: {origin_fairness["Qwen"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Awareness of Fairness', fontsize=13)
    ax.set_title('Qwen2-7B-Instruct', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_fairness['Qwen']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.set_xlabel('Method', fontsize=13)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)

    # Mistral Fairness
    ax = axes[0, 1]
    methods_list = [m for m in METHODS if m in fairness_data and 'Mistral' in fairness_data[m]]
    values = [fairness_data[m]['Mistral'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_fairness['Mistral'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_fairness['Mistral'] + 0.01, f'Origin Model: {origin_fairness["Mistral"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_title('Mistral-7B-Instruct-v0.2', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_fairness['Mistral']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_ylabel('Awareness of Fairness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)

    # Vicuna Fairness
    ax = axes[0, 2]
    methods_list = [m for m in METHODS if m in fairness_data and 'Vicuna' in fairness_data[m]]
    values = [fairness_data[m]['Vicuna'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_fairness['Vicuna'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_fairness['Vicuna'] + 0.01, f'Origin Model: {origin_fairness["Vicuna"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_title('Vicuna-7B-v1.5', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_fairness['Vicuna']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_ylabel('Awareness of Fairness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)

    # ============= Privacy Awareness (Bottom Row) =============

    # Qwen Privacy
    ax = axes[1, 0]
    methods_list = [m for m in METHODS if m in privacy_data and 'Qwen' in privacy_data[m]]
    values = [privacy_data[m]['Qwen'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_privacy['Qwen'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_privacy['Qwen'] + 0.01, f'Origin Model: {origin_privacy["Qwen"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Awareness of Privacy', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_privacy['Qwen']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Qwen2-7B-Instruct', fontsize=14, pad=10)

    # Mistral Privacy
    ax = axes[1, 1]
    methods_list = [m for m in METHODS if m in privacy_data and 'Mistral' in privacy_data[m]]
    values = [privacy_data[m]['Mistral'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_privacy['Mistral'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_privacy['Mistral'] + 0.01, f'Origin Model: {origin_privacy["Mistral"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Awareness of Privacy', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_privacy['Mistral']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Mistral-7B-Instruct-v0.2', fontsize=14, pad=10)

    # Vicuna Privacy
    ax = axes[1, 2]
    methods_list = [m for m in METHODS if m in privacy_data and 'Vicuna' in privacy_data[m]]
    values = [privacy_data[m]['Vicuna'] for m in methods_list]
    bars = ax.bar(
        methods_list,
        values,
        color=[COLORS[m] for m in methods_list],
        edgecolor='none',
        width=0.58,
    )
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{height:.4f}',
            ha='center',
            va='bottom',
            fontsize=12,
        )
    
    # Add origin baseline line
    ax.axhline(origin_privacy['Vicuna'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_privacy['Vicuna'] + 0.01, f'Origin Model: {origin_privacy["Vicuna"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Awareness of Privacy', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_privacy['Vicuna']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Vicuna-7B-v1.5', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig('/home/scur0168/fact-ai-group-23/reproduction/artifacts/malicious-data/malicious_awareness_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/scur0168/fact-ai-group-23/reproduction/artifacts/malicious-data/malicious_awareness_comparison.pdf', bbox_inches='tight')
    print("✓ Plots saved: /home/scur0168/fact-ai-group-23/reproduction/artifacts/malicious-data/malicious_awareness_comparison.png and malicious_awareness_comparison.pdf")
    
    plt.show()

if __name__ == '__main__':
    plot_bar_charts()
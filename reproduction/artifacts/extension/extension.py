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
MODELS = ['Qwen', 'Mistral', 'Vicuna']
MODEL_TITLES = {
    'Qwen': 'Qwen2-7B-Instruct',
    'Mistral': 'Mistral-7B-Instruct-v0.2',
    'Vicuna': 'Vicuna-7B-v1.5',
}

OUTPUT_DIR = Path(__file__).parent

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

def get_matrix(metric_type):
    """Return a matrix (methods x models) of awareness rates."""
    data = extract_values(metric_type)
    matrix = []
    for method in METHODS:
        row = []
        for model in MODELS:
            row.append(data[method][model])
        matrix.append(row)
    return matrix


def plot_bar_charts(show=False):
    """Create side-by-side bar charts for safety and helpfulness awareness (original style)."""
    
    # Extract data
    safety_data = extract_values('safety')
    helpfulness_data = extract_values('helpfulness')
    origin_safety = extract_origin('safety')
    origin_helpfulness = extract_origin('helpfulness')
    
    # Create figure with 2x3 subplots (2 rows: safety and helpfulness, 3 cols: Qwen, Mistral, Vicuna)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    
    # ============= Safety Awareness (Top Row) =============

    # Qwen Safety
    ax = axes[0, 0]
    methods_list = [m for m in METHODS if m in safety_data and 'Qwen' in safety_data[m]]
    values = [safety_data[m]['Qwen'] for m in methods_list]
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
    ax.axhline(origin_safety['Qwen'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_safety['Qwen'] + 0.01, f'Origin Model: {origin_safety["Qwen"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Safety Awareness', fontsize=13)
    ax.set_title('Qwen2-7B-Instruct', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_safety['Qwen']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.set_xlabel('Method', fontsize=13)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)

    # Mistral Safety
    ax = axes[0, 1]
    methods_list = [m for m in METHODS if m in safety_data and 'Mistral' in safety_data[m]]
    values = [safety_data[m]['Mistral'] for m in methods_list]
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
    ax.axhline(origin_safety['Mistral'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_safety['Mistral'] + 0.01, f'Origin Model: {origin_safety["Mistral"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_title('Mistral-7B-Instruct-v0.2', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_safety['Mistral']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_ylabel('Safety Awareness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)

    # Vicuna Safety
    ax = axes[0, 2]
    methods_list = [m for m in METHODS if m in safety_data and 'Vicuna' in safety_data[m]]
    values = [safety_data[m]['Vicuna'] for m in methods_list]
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
    ax.axhline(origin_safety['Vicuna'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_safety['Vicuna'] + 0.01, f'Origin Model: {origin_safety["Vicuna"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_title('Vicuna-7B-v1.5', fontsize=14, pad=10)
    ax.set_ylim(0, max(values + [origin_safety['Vicuna']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_ylabel('Safety Awareness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)

    # ============= Helpfulness Awareness (Bottom Row) =============

    # Qwen Helpfulness
    ax = axes[1, 0]
    methods_list = [m for m in METHODS if m in helpfulness_data and 'Qwen' in helpfulness_data[m]]
    values = [helpfulness_data[m]['Qwen'] for m in methods_list]
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
    ax.axhline(origin_helpfulness['Qwen'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_helpfulness['Qwen'] + 0.01, f'Origin Model: {origin_helpfulness["Qwen"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Helpfulness Awareness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_helpfulness['Qwen']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Qwen2-7B-Instruct', fontsize=14, pad=10)

    # Mistral Helpfulness
    ax = axes[1, 1]
    methods_list = [m for m in METHODS if m in helpfulness_data and 'Mistral' in helpfulness_data[m]]
    values = [helpfulness_data[m]['Mistral'] for m in methods_list]
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
    ax.axhline(origin_helpfulness['Mistral'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_helpfulness['Mistral'] + 0.01, f'Origin Model: {origin_helpfulness["Mistral"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Helpfulness Awareness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_helpfulness['Mistral']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Mistral-7B-Instruct-v0.2', fontsize=14, pad=10)

    # Vicuna Helpfulness
    ax = axes[1, 2]
    methods_list = [m for m in METHODS if m in helpfulness_data and 'Vicuna' in helpfulness_data[m]]
    values = [helpfulness_data[m]['Vicuna'] for m in methods_list]
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
    ax.axhline(origin_helpfulness['Vicuna'], color='gray', linestyle='--', linewidth=1.5)
    ax.text(0.15, origin_helpfulness['Vicuna'] + 0.01, f'Origin Model: {origin_helpfulness["Vicuna"]:.4f}',
            fontsize=11, color='dimgray', va='bottom')

    ax.set_ylabel('Helpfulness Awareness', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylim(0, max(values + [origin_helpfulness['Vicuna']]) + 0.08)
    ax.set_xticks(range(len(methods_list)))
    ax.set_xticklabels(methods_list, fontsize=12)
    ax.grid(axis='y', alpha=0.35, linestyle='-')
    ax.set_axisbelow(True)
    ax.set_title('Vicuna-7B-v1.5', fontsize=14, pad=10)
    
    plt.tight_layout()
    
    # Save figures
    plt.savefig('/home/scur0168/fact-ai-group-23/reproduction/artifacts/extension/extension.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/scur0168/fact-ai-group-23/reproduction/artifacts/extension/extension.pdf', bbox_inches='tight')
    print("✓ Plots saved: /home/scur0168/fact-ai-group-23/reproduction/artifacts/extension/extension.png and extension.pdf")
    
    if show:
        plt.show()


def plot_bar_charts_clean():
    """Cleaner small-multiples bar chart with shared legend and fewer labels."""
    safety_data = extract_values('safety')
    helpfulness_data = extract_values('helpfulness')
    origin_safety = extract_origin('safety')
    origin_helpfulness = extract_origin('helpfulness')

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))

    def plot_row(row_idx, metric_name, data, origin):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            methods_list = [m for m in METHODS if m in data and model in data[m]]
            values = [data[m][model] for m in methods_list]
            ax.bar(
                methods_list,
                values,
                color=[COLORS[m] for m in methods_list],
                edgecolor='none',
                width=0.6,
            )
            ax.axhline(origin[model], color='gray', linestyle='--', linewidth=1.0, alpha=0.8)
            ax.set_title(MODEL_TITLES[model], fontsize=12, pad=8)
            ax.set_ylim(0, max(values + [origin[model]]) + 0.08)
            ax.grid(axis='y', alpha=0.25, linestyle='-')
            ax.set_axisbelow(True)
            if col_idx == 0:
                ax.set_ylabel(f'{metric_name} Awareness', fontsize=12)
            ax.set_xlabel('Method', fontsize=11)
            ax.tick_params(axis='x', labelrotation=15, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

    plot_row(0, 'Safety', safety_data, origin_safety)
    plot_row(1, 'Helpfulness', helpfulness_data, origin_helpfulness)

    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=6) for m in METHODS]
    labels = METHODS
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'awareness_bars_clean.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'awareness_bars_clean.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_heatmaps():
    """Two heatmaps (safety/helpfulness) with methods x models."""
    safety = get_matrix('safety')
    helpfulness = get_matrix('helpfulness')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, data, title in zip(axes, [safety, helpfulness], ['Safety Awareness', 'Helpfulness Awareness']):
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_TITLES[m] for m in MODELS], rotation=20, ha='right', fontsize=9)
        ax.set_yticks(range(len(METHODS)))
        ax.set_yticklabels(METHODS, fontsize=10)
        ax.set_title(title, fontsize=12, pad=8)
        for i in range(len(METHODS)):
            for j in range(len(MODELS)):
                ax.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', fontsize=9, color='white')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'awareness_heatmaps.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'awareness_heatmaps.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_tradeoff_scatter():
    """Safety vs helpfulness scatter (one subplot per model)."""
    safety_data = extract_values('safety')
    helpfulness_data = extract_values('helpfulness')
    origin_safety = extract_origin('safety')
    origin_helpfulness = extract_origin('helpfulness')

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, model in zip(axes, MODELS):
        ox = origin_helpfulness[model]
        oy = origin_safety[model]
        xs = [ox] + [helpfulness_data[method][model] for method in METHODS]
        ys = [oy] + [safety_data[method][model] for method in METHODS]
        ax.scatter(
            ox,
            oy,
            s=140,
            marker='D',
            color='#7ED487',
            edgecolor='none',
            zorder=3,
        )
        ax.text(ox, oy + 0.015, 'Origin', fontsize=8, ha='center', va='bottom')

        for method in METHODS:
            x = helpfulness_data[method][model]
            y = safety_data[method][model]
            ax.scatter(
                x,
                y,
                s=140,
                marker='o',
                color=COLORS[method],
                edgecolor='none',
                zorder=2,
            )
            ax.text(x, y + 0.015, f'{method}', fontsize=8, ha='center', va='bottom')

        ax.set_title(MODEL_TITLES[model], fontsize=11, pad=6)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max(0.02, (x_max - x_min) * 0.12)
        y_pad = max(0.02, (y_max - y_min) * 0.12)
        ax.set_xlim(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad))
        ax.set_ylim(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad))
        ax.set_xlabel('Awareness of Helpfulness', fontsize=10)

    axes[0].set_ylabel('Awareness of Safety', fontsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'awareness_tradeoff_scatter.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'awareness_tradeoff_scatter.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_radar():
    """Radar charts across models for each method (two panels: safety/helpfulness)."""
    safety = get_matrix('safety')
    helpfulness = get_matrix('helpfulness')

    angles = [n / float(len(MODELS)) * 2 * 3.14159265 for n in range(len(MODELS))]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, subplot_kw=dict(polar=True), figsize=(10, 4.5))
    for ax, data, title in zip(axes, [safety, helpfulness], ['Safety Awareness', 'Helpfulness Awareness']):
        ax.set_theta_offset(3.14159265 / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([MODEL_TITLES[m] for m in MODELS], fontsize=9)
        ax.set_rlabel_position(0)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=8)
        ax.set_title(title, fontsize=12, pad=12)
        for method_idx, method in enumerate(METHODS):
            values = data[method_idx] + [data[method_idx][0]]
            ax.plot(angles, values, color=COLORS[method], linewidth=1.8, label=method)
            ax.fill(angles, values, color=COLORS[method], alpha=0.12)

    axes[0].legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'awareness_radar.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'awareness_radar.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_grouped_by_model():
    """Grouped bars per model, one panel per metric (like the reference image)."""
    safety_data = extract_values('safety')
    helpfulness_data = extract_values('helpfulness')
    origin_safety = extract_origin('safety')
    origin_helpfulness = extract_origin('helpfulness')

    metrics = [
        ('Awareness of Safety', safety_data, origin_safety),
        ('Awareness of Helpfulness', helpfulness_data, origin_helpfulness),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    bar_width = 0.13
    x = list(range(len(MODELS)))
    offsets = [(-1.5 + i) * bar_width for i in range(len(METHODS))]
    xlabels = ["qwen2", "mistral", "vicuna"]

    for ax, (title, data, origin) in zip(axes, metrics):
        max_val = 0.0
        for idx, method in enumerate(METHODS):
            values = [data[method][model] for model in MODELS]
            max_val = max(max_val, max(values))
            positions = [xi + offsets[idx] for xi in x]
            hatch = '///' if method == 'DoRA' else None
            ax.bar(
                positions,
                values,
                width=bar_width,
                color=COLORS[method],
                edgecolor='none',
                label=method,
                hatch=hatch,
            )
        origin_mean = sum(origin[m] for m in MODELS) / len(MODELS)
        ax.axhline(origin_mean, color='gray', linestyle='--', linewidth=1.2, label='Origin')
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Awareness Rate', fontsize=9)
        ax.grid(axis='y', alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, min(1.0, max_val + 0.1))

    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=6) for m in METHODS]
    handles.append(plt.Line2D([0], [0], color='gray', lw=1.2, linestyle='--'))
    labels = METHODS + ['Origin']
    fig.legend(handles, labels, ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'awareness_grouped_by_model.png', dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT_DIR / 'awareness_grouped_by_model.pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    plot_bar_charts()
    plot_bar_charts_clean()
    plot_heatmaps()
    plot_tradeoff_scatter()
    plot_radar()
    plot_grouped_by_model()

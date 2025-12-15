import matplotlib.pyplot as plt

# Set academic style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})

# ==========================================
# FIGURE 1: Evolutionary Convergence
# ==========================================
def plot_fitness():
    # Data from Appendix A.2
    generations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    mean_fitness = [0.523, 0.541, 0.558, 0.572, 0.584, 0.593, 0.601, 0.608, 0.614, 0.619, 0.623, 0.631, 0.635]
    best_fitness = [0.672, 0.689, 0.701, 0.715, 0.728, 0.739, 0.748, 0.756, 0.763, 0.769, 0.774, 0.782, 0.787]
    diversity = [0.342, 0.335, 0.328, 0.321, 0.315, 0.309, 0.304, 0.299, 0.295, 0.291, 0.288, 0.281, 0.278]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Fitness (Left Axis)
    ax1.plot(generations, best_fitness, 'o-', color='#2E86C1', label='Best Fitness', linewidth=2, markersize=5)
    ax1.plot(generations, mean_fitness, 's--', color='#2874A6', label='Mean Fitness', linewidth=2, markersize=5)
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Fitness Score (Normalized)')
    ax1.set_ylim(0.4, 0.9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot Diversity (Right Axis)
    ax2 = ax1.twinx()
    ax2.plot(generations, diversity, 'v:', color='#E74C3C', label='Population Diversity', linewidth=1.5, markersize=5)
    ax2.set_ylabel('Diversity Index', color='#E74C3C')
    ax2.tick_params(axis='y', labelcolor='#E74C3C')
    ax2.set_ylim(0, 0.5)

    # Combined Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

    plt.title('Evolutionary Optimization: Fitness Convergence & Diversity Retention')
    plt.tight_layout()
    plt.savefig('fig1_fitness.png', dpi=300)
    print("Generated fig1_fitness.png")

# ==========================================
# FIGURE 2: Latency Performance
# ==========================================
def plot_latency():
    # Data from Appendix A.1
    labels = ['Genetic Crossover', 'LNN (10 seq)', 'LNN (50 seq)', 'LNN (100 seq)']
    times_ms = [2.3, 2.1, 8.7, 16.3]
    errors = [0.5, 0.3, 1.2, 2.1]  # Standard Deviation

    fig, ax = plt.subplots(figsize=(8, 5))

    # Create Bar Chart
    bars = ax.bar(labels, times_ms, yerr=errors, capsize=5, color=['#27AE60', '#D35400', '#D35400', '#C0392B'], alpha=0.8)

    # Add Value Labels on top
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Target Line (<10ms soft real-time limit)
    ax.axhline(y=10, color='gray', linestyle='--', linewidth=1, label='Real-time Threshold (10ms)')

    ax.set_ylabel('Processing Latency (ms)')
    ax.set_title('System Latency: Genetic Ops vs. LNN Sequence Lengths')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('fig2_latency.png', dpi=300)
    print("Generated fig2_latency.png")

if __name__ == "__main__":
    plot_fitness()
    plot_latency()

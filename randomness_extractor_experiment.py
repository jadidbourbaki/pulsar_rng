from pulsar import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# python3 pulsar.py rng -q gray_coding -dm none -n J0030+0451 -d EPTA -e
ent1 = 4.640000

# python3 pulsar.py rng -q gray_coding -dm xor -n J0030+0451 -d EPTA -e
ent2 = 5.116569

# pulsar.py rng -q gray_coding -dm von_neumann -n J0030+0451 -d EPTA -e
ent3 = 6.664750

# python3 pulsar.py rng -q gray_coding -dm shake256 -n J0030+0451 -d EPTA -e
ent4 = 7.948123

# We investigate four randomness extractors:
# - None
# - XOR 
# - Von Neumann
# - Shake-256
randomness_extractor = ['None', 'XOR', 'Von Neumann', 'SHAKE-256']

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.25  # Increase bar width to accommodate all bars in one set
x = np.array([0, 1, 2, 3])  # X positions for each quantification method

entropy = [ent1, ent2, ent3, ent4]
# Bar patterns and colors
patterns = ['//', '\\\\', '||', '--']  
colors = ['lightgray', 'darkgray', 'black', 'dimgray']

# Create a single set of bars
bars = ax.bar(x, entropy, width=bar_width, edgecolor='black', color=colors, hatch=patterns)

# Axis labels and ticks
ax.set_ylabel('Entropy (bits per byte)', fontsize=22, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(randomness_extractor, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 10)

# Grid styling
ax.grid(True, axis='y', linestyle='-', linewidth=0.6, alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, f"{height:.3f}", ha='center', va='bottom', fontsize=16)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'{PLOT_DATA_FOLDER}/randomness_extractor_entropy_bar_chart.pdf', format='pdf', dpi=600)

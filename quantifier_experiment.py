from pulsar import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# python3 pulsar.py rng -d EPTA -n J0030+0451 -dm none -q threshold -e
ent1 = 3.530587

# python3 pulsar.py rng -d EPTA -n J0030+0451 -dm none -q gray_coding -e
ent2 = 4.640000

# python3 pulsar.py rng -d EPTA -n J0030+0451 -dm none -q sha512 -e 
ent3 = 7.998709

# python3 pulsar.py rng -d NANOGrav -n J0030+0451 -dm none -q threshold -e
ent4 = 3.510270

# python3 pulsar.py rng -d NANOGrav -n J0030+0451 -dm none -q gray_coding -e
ent5 = 6.804345

# python3 pulsar.py rng -d NANOGrav -n J0030+0451 -dm none -q sha512 -e
ent6 = 7.998553

# python3 pulsar.py rng -d EPTA -n J1918-0642 -dm none -q threshold -e
ent7 = 1.079481

# python3 pulsar.py rng -d EPTA -n J1918-0642 -dm none -q gray_coding -e
ent8 = 5.584438

# python3 pulsar.py rng -d EPTA -n J1918-0642 -dm none -q sha512 -e
ent9 = 7.998481

# python3 pulsar.py rng -d NANOGrav -n J1918-0642 -dm none -q threshold -e
ent10 = 6.945456

# python3 pulsar.py rng -d NANOGrav -n J1918-0642 -dm none -q gray_coding -e
ent11 = 5.265700

# python3 pulsar.py rng -d NANOGrav -n J1918-0642 -dm none -q sha512 -e
ent12 = 7.998358

# entropy in bits per byte
entropy = [ent1, ent4, ent7, ent10, ent2, ent5, ent8, ent11, ent3, ent6, ent9, ent12]

# We investigate three quantification techniques on the normalized residual values: 
# - using a simple threshold ($\tau = 0.5$), 
# - 8-bit Gray coding, and 
# - using the 8-bit Gray coded value as a seed for a SHA-512 hash.
quantification_method = ['Threshold', '8-bit Gray Coding', 'SHA-512']
pulsars = ['J0030+0451 (EPTA)', 'J0030+0451 (NANOGrav)', 'J1918-0642 (EPTA)', 'J1918-0642 (NANOGrav)']

# Create the bar chart
fig, ax = plt.subplots(figsize=(12, 7))

bar_width = 0.15
x = np.array([0, 1, 2])  # X positions for each quantification method
offsets = np.linspace(-1.5 * 0.2, 1.5 * 0.2, 4)  # Offsets for different pulsars

# Bar patterns and colors
patterns = ['//', '\\\\', '||', '--']  
colors = ['lightgray', 'darkgray', 'black', 'dimgray']

bars = []
for i, (offset, pattern, color) in enumerate(zip(offsets, patterns, colors)):
    bars.append(ax.bar(x + offset, entropy[i::4], width=bar_width, hatch=pattern, edgecolor='black', color=color, label=pulsars[i]))

# Axis labels and ticks
ax.set_ylabel('Entropy (bits per byte)', fontsize=22, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(quantification_method, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0, 10)

# Grid styling
ax.grid(True, axis='y', linestyle='-', linewidth=0.6, alpha=0.7)

# Add value labels on top of bars
for bar_group in bars:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.2, f"{height:.3f}", ha='center', va='bottom', fontsize=16)

# Legend improvements
legend_patches = [mpatches.Patch(facecolor=color, edgecolor='black', hatch=pattern, label=label)
                  for color, pattern, label in zip(colors, patterns, pulsars)]
ax.legend(handles=legend_patches, fontsize=18, loc='upper left', frameon=False)

# Adjust layout and save
plt.tight_layout()
plt.savefig(f'{PLOT_DATA_FOLDER}/entropy_bar_chart.pdf', format='pdf', dpi=600)
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Define file paths
SAVE_DIR = '../SoccerNet/SN-BAS-2025-SAVE_DIR'
stats_path = os.path.join(SAVE_DIR, 'action_counts.json')

# Read action counts statistics from JSON
with open(stats_path, 'r') as f:
    data = json.load(f)

# Define colors for each split
colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

# Generate a plot for each split
for split, color in colors.items():
    if split not in data:
        print(f"Warning: '{split}' not found in the JSON data.")
        continue
        
    histogram = data[split]["histogram"]
    x_labels = np.arange(len(histogram))
    counts = histogram
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(x_labels))
    
    ax.bar(x, counts, color=color, edgecolor='black', width=0.6)
    
    # Formatting labels and title
    ax.set_xlabel('# of Actions', fontsize=12)
    ax.set_ylabel('# of Clips', fontsize=12)
    
    total_clips = sum(counts)
    ax.set_title(f'Actions per Clip Histogram ({split} split - {total_clips} total clips)', fontsize=15, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    
    # Add value labels on top of each bar for clarity
    max_count = max(counts) if counts else 1
    for i, v in enumerate(counts):
        ax.text(i, v + (max_count * 0.01), str(v), ha='center', va='bottom', fontsize=11)
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # OPTIONAL: Uncomment the next line if the '0' bar is so huge that it makes the other bars invisible
    # ax.set_yscale('log') 
    
    fig.tight_layout()
    plot_filename = os.path.join(SAVE_DIR, f'clip_actions_{split}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved as '{plot_filename}'")
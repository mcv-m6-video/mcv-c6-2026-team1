import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Define file paths
SAVE_DIR = '../SoccerNet/SN-BAS-2025-SAVE_DIR'
class_txt_path = 'data/soccernetball/class.txt'

# Read ordered classes from class.txt
with open(class_txt_path, 'r') as f:
    ordered_classes = ["BACKGROUND"] + [line.strip() for line in f.readlines() if line.strip()]

# Read dataset statistics from JSON
with open(os.path.join(SAVE_DIR, 'dataset_statistics.json'), 'r') as f:
    data = json.load(f)

# Define colors for each split
colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}

# Generate a plot for each split
for split, color in colors.items():
    if split not in data:
        print(f"Warning: '{split}' not found in the JSON data.")
        continue
        
    split_data = data[split]
    total_frames = split_data["total_frames"]
    counts = [split_data["counts"][cat] for cat in ordered_classes]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(ordered_classes))
    
    ax.bar(x, counts, color=color, edgecolor='black')
    ax.set_ylabel('# samples', fontsize=12)
    ax.set_title(f'Dataset Statistics ({split} split - {total_frames} total frames)', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_classes, rotation=25, ha='right', fontsize=10)
    
    # Add value labels on top of each bar for clarity
    max_count = max(counts) if counts else 1
    for i, v in enumerate(counts):
        ax.text(i, v + (max_count * 0.01), str(v), ha='center', va='bottom', fontsize=10)
        
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    plot_filename = os.path.join(SAVE_DIR, f'dataset_statistics_{split}.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Plot saved as '{plot_filename}'")
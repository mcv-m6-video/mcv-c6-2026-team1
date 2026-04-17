import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data
data = {
    "RegNet-Y (200MF)": {"MACs": 18.19,  "AP10": 28.94, "Params": 4.41},
    "RegNet-Y (400MF)": {"MACs": 36.31,  "AP10": 32.98, "Params": 5.67},
    "RegNet-Y (800MF)": {"MACs": 71.58,  "AP10": 34.55, "Params": 7.99},
    "ViT (base)":         {"MACs": 843.27, "AP10": 29.51, "Params": 88.17},
}

# Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.linewidth": 1.0,
})

# Extract values
experiments = list(data.keys())
macs = np.array([data[e]["MACs"] for e in experiments], dtype=float)
ap10 = np.array([data[e]["AP10"] for e in experiments], dtype=float)
params = np.array([data[e]["Params"] for e in experiments], dtype=float)

# X-axis
# log-scale MACs, then normalize
log_macs = np.log10(macs)
x_min, x_max = 0.8, 9.2
macs_plot = x_min + (log_macs - log_macs.min()) / (log_macs.max() - log_macs.min()) * (x_max - x_min)

# Bubble sizes
size_scale = 380
bubble_sizes = np.sqrt(params) * size_scale

colors = {
    "RegNet-Y (200MF)": "#4C78A8",
    "RegNet-Y (400MF)": "#8FA9C9",
    "RegNet-Y (800MF)": "#F28E2B",
    "ViT (base)": "#EDC08B",
}

# Choose which model to emphasize
best_model = "RegNet-Y (800MF)"

# Plot
fig, ax = plt.subplots(figsize=(10.5, 6.2))

for i, exp in enumerate(experiments):
    is_best = (exp == best_model)

    ax.scatter(
        macs_plot[i],
        ap10[i],
        s=bubble_sizes[i],
        color=colors[exp],
        alpha=0.72 if is_best else 0.55,
        edgecolors="#8B0000" if is_best else "#333333",
        linewidths=2.0 if is_best else 1.0,
        zorder=4 if is_best else 3
    )

# Labels outside bubbles
label_offsets = {
    "RegNet-Y (200MF)": (18, 0),
    "RegNet-Y (400MF)": (20, 0),
    "RegNet-Y (800MF)": (20, 0),
    "ViT (base)": (-10, -45),
}

for i, exp in enumerate(experiments):
    dx, dy = label_offsets[exp]
    ax.annotate(
        exp,
        (macs_plot[i], ap10[i]),
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold" if exp == best_model else "normal",
        color="#111111"
    )

# Titles and labels
ax.set_xlabel("GMACs (log-scale)", labelpad=18)
ax.set_ylabel("mAP@10", labelpad=8)

# Ticks: show real MAC values at normalized positions
ax.set_xticks(macs_plot)
ax.set_xticklabels([f"{m:.0f}" if m >= 100 else f"{m:.2f}" for m in macs])

# Grid / limits / spines
ax.set_ylim(25, 38)
ax.set_xlim(0, 10)

ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.22)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("bubble_ap10_macs_final.png", bbox_inches="tight")
plt.show()
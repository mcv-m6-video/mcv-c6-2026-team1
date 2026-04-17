import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


common_baseline = 8.08
metric_name = "mAP@10"
best_score = 35.43

best_configs = [
    "mAP@10", "GRU", "RegNet-Y (800MF)", "Focal Loss", "clip_len 50", "stride 2"
]

cursive_exp = {
    "clip_len 25", "clip_len 50", "clip_len 100", "stride 1", "stride 2", "stride 4"
}

experiments = {
    "1": {
        "previous_best": 8.08,
        "ablations": {
            "mAP@10": 10.10
        },
    },
    "2": {
        "previous_best": 10.10,
        "ablations": {
            "Transformer": 17.21,
            "GRU": 28.94
        },
    },
    "3": {
        "previous_best": 28.94,
        "ablations": {
            "RegNet-Y (200MF)": 28.94,
            "RegNet-Y (400MF)": 32.98,
            "RegNet-Y (800MF)": 34.55,
            "ViT (base)": 29.51
        },
    },
    "4": {
        "previous_best": 34.55,
        "ablations": {
            "Dynamic Weights": 31.29,
            "Focal Loss": 35.43,
            "Oversampling": 34.19,
        },
    },
    "5": {
        "previous_best": 35.43,
        "ablations": {
            "clip_len 25": 31.57,
            "clip_len 50": 35.43,
            "clip_len 100": 33.92,
            "stride 1": 31.66, 
            "stride 2": 35.43,
            "stride 4": 34.23
        },
    },
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})

def plot_floating_ablation_bars_compact(
    experiments,
    common_baseline,
    metric_name="Metric",
    save_path="floating_ablation_bars_compact.png"
):
    fig, ax = plt.subplots(figsize=(10.8, 5.8))

    x_positions = []
    x_ticklabels = []
    group_centers = []
    group_labels = []
    group_boundaries = []

    current_x = 0
    group_gap = 0.9
    intra_gap = 0.95
    bar_width = 0.62

    # Baseline
    baseline_x = current_x
    ax.bar(
        baseline_x,
        common_baseline,
        width=bar_width,
        color="steelblue",
        edgecolor="black",
        linewidth=0.8,
        zorder=3
    )
    ax.text(
        baseline_x,
        common_baseline + 0.10,
        f"{common_baseline:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

    x_positions.append(baseline_x)
    x_ticklabels.append("Baseline")

    current_x += 1.2 + group_gap

    # Experiment groups
    for exp_name, exp_data in experiments.items():
        prev_best = exp_data["previous_best"]
        ablation_names = list(exp_data["ablations"].keys())
        ablation_scores = list(exp_data["ablations"].values())

        n = len(ablation_names)
        xs = current_x + np.arange(n) * intra_gap
        center = xs.mean()

        # store boundaries for separators and title placement
        left_boundary = xs[0] - 0.55
        right_boundary = xs[-1] + 0.55
        group_boundaries.append((left_boundary, right_boundary))

        # dashed line for previous best across the group
        ax.hlines(
            y=prev_best,
            xmin=xs[0] - 0.40,
            xmax=xs[-1] + 0.40,
            colors="black",
            linestyles="--",
            linewidth=1.2,
            zorder=2
        )

        for x, ab_name, score in zip(xs, ablation_names, ablation_scores):
            delta = score - prev_best

            if delta > 0:
                bottom = prev_best
                height = delta
                color_bar = "#2ca02c"
                delta_y = score + 1.35
                delta_va = "bottom"
            elif delta == 0:
                bottom = prev_best
                height = delta
                color_bar = "#f79503"
                delta_y = score + 1.35
                delta_va = "bottom"
            else:
                bottom = score
                height = -delta
                color_bar = "#d62728"
                delta_y = score - 1.4
                delta_va = "top"

            ax.bar(
                x,
                height,
                bottom=bottom,
                width=bar_width,
                color=color_bar,
                edgecolor="black",
                linewidth=0.7,
                zorder=3
            )

            # absolute score
            if score != best_score:
                ax.text(
                    x,
                    score + 0.13 if delta >= 0 else score - 0.2,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=9,
                    color="black"
                )
            else:
                ax.text(
                    x,
                    score + 0.13 if delta >= 0 else score - 0.2,
                    f"{score:.2f}",
                    ha="center",
                    va="bottom" if delta >= 0 else "top",
                    fontsize=9,
                    color="black",
                    fontweight="bold"
                )
                

            # delta outside the bar
            delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
            ax.text(
                x,
                delta_y,
                delta_str,
                ha="center",
                va=delta_va,
                fontsize=9,
                fontweight="bold",
                color=color_bar
            )

            x_positions.append(x)
            x_ticklabels.append(ab_name)

        group_centers.append(center)
        group_labels.append(exp_name)

        current_x = xs[-1] + intra_gap + group_gap

    # X ticks
    ax.set_xticks(x_positions)

    tick_labels = ax.set_xticklabels(x_ticklabels, rotation=28, ha="right")

    for label in tick_labels:
        if label.get_text() in cursive_exp:
            label.set_fontstyle("italic")
    for label in tick_labels:
        if label.get_text() in best_configs:
            label.set_fontweight("bold")
    ax.set_xticklabels(x_ticklabels, rotation=28, ha="right")


    ax.set_ylabel(metric_name)

    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # tighter x-limits to reduce white space
    ax.set_xlim(min(x_positions) - 0.8, max(x_positions) + 0.6)

    all_scores = [common_baseline]
    for exp_data in experiments.values():
        all_scores.append(exp_data["previous_best"])
        all_scores.extend(exp_data["ablations"].values())

    ax.set_ylim(min(all_scores) - 0.45, max(all_scores) + 0.9)


    # Group titles above the groups
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for center, label in zip(group_centers, group_labels):
        ax.text(
            center,
            1.065,                   # slightly above the axes
            label,
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            color="0.2",
            clip_on=False
        )

    # Light vertical separators between groups
    for i in range(len(group_boundaries) - 1):
        _, right_current = group_boundaries[i]
        left_next, _ = group_boundaries[i + 1]
        separator_x = (right_current + left_next) / 2.0
        ax.axvline(
            x=separator_x,
            color="0.85",
            linewidth=1.0,
            zorder=1
        )

    # Legend
    legend_items = [
        Patch(facecolor="steelblue", edgecolor="black", label="Common baseline"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="Better than previous best"),
        Patch(facecolor="#d62728", edgecolor="black", label="Worse than previous best"),
        Line2D([0], [0], color="black", linestyle="--", label="Previous best"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.03),
        frameon=True
    )

    plt.subplots_adjust(bottom=0.18, left=0.08, right=0.98, top=0.85)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


plot_floating_ablation_bars_compact(
    experiments,
    common_baseline=common_baseline,
    metric_name=metric_name,
    save_path="floating_ablation_bars_compact.png"
)
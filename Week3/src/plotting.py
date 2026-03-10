import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

SAVE_DIR = Path("./results")


def plot_flow_metrics(msen, pepn, methods, save_path=None):
    """
    Create grouped bar plot comparing MSEN and PEPN for optical flow methods.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7,4))

    bars1 = ax.bar(x - width/2, msen, width, label="MSEN")
    bars2 = ax.bar(x + width/2, pepn, width, label="PEPN")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Error")
    ax.set_title("Optical Flow Performance Comparison")
    ax.legend()

    # add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")



def plot_flow_metrics_cvpr(
    methods,
    msen,
    pepn,
    times=None,
    title="Optical Flow Benchmark",
    save_path=None,
):
    """
    Publication-style grouped bar plot for optical flow metrics.

    Parameters
    ----------
    methods : list[str]
        Method names.
    msen : list[float]
        MSEN values (lower is better).
    pepn : list[float]
        PEPN values (lower is better).
    times : list[float] | None
        Optional runtimes in seconds. If given, shown under x tick labels.
    title : str
        Plot title.
    save_path : str | None
        If provided, saves the figure.
    """
    methods = list(methods)
    msen = np.asarray(msen, dtype=float)
    pepn = np.asarray(pepn, dtype=float)

    if len(methods) != len(msen) or len(methods) != len(pepn):
        raise ValueError("methods, msen, and pepn must have the same length.")

    if times is not None:
        times = np.asarray(times, dtype=float)
        if len(times) != len(methods):
            raise ValueError("times must have the same length as methods.")

    # Clean labels
    xticklabels = []
    for i, m in enumerate(methods):
        if times is None:
            xticklabels.append(m)
        else:
            xticklabels.append(f"{m}\n{times[i]:.2f}s")

    # Best methods
    best_msen_idx = int(np.argmin(msen))
    best_pepn_idx = int(np.argmin(pepn))

    # Figure setup
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(methods))
    width = 0.34

    # Muted, paper-friendly colors
    c_msen = "#4C78A8"
    c_pepn = "#F58518"
    c_best_edge = "#111111"

    bars_msen = ax.bar(
        x - width / 2,
        msen,
        width,
        label="MSEN",
        color=c_msen,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    bars_pepn = ax.bar(
        x + width / 2,
        pepn,
        width,
        label="PEPN",
        color=c_pepn,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # Highlight best bars with thicker edge
    bars_msen[best_msen_idx].set_linewidth(2.0)
    bars_msen[best_msen_idx].set_edgecolor(c_best_edge)

    bars_pepn[best_pepn_idx].set_linewidth(2.0)
    bars_pepn[best_pepn_idx].set_edgecolor(c_best_edge)

    # Value labels
    y_max = max(float(msen.max()), float(pepn.max()))
    label_offset = 0.02 * y_max

    for bar in bars_msen:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars_pepn:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + label_offset,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Axes and grid
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Error")
    ax.set_title(title, pad=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)

    # Give headroom for labels
    ax.set_ylim(0, y_max * 1.18)

    # Legend with note for best bars
    legend_handles = [
        Patch(facecolor=c_msen, edgecolor="black", label="MSEN"),
        Patch(facecolor=c_pepn, edgecolor="black", label="PEPN"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    # Small annotation for best method(s)
    annotation_lines = [
        f"Best MSEN: {methods[best_msen_idx]} ({msen[best_msen_idx]:.2f})",
        f"Best PEPN: {methods[best_pepn_idx]} ({pepn[best_pepn_idx]:.2f})",
    ]
    ax.text(
        0.01,
        0.98,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
    )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")


if __name__=="__main__":
    methods = ["PyFlow slow", "PyFlow fast", "GMFlow", "MemFlow"]

    msen = [0.94, 1.04, 0.72, 0.53]
    pepn = [7.43, 8.69, 2.77, 2.63]

    plot_flow_metrics(
        msen,
        pepn,
        methods,
        save_path=SAVE_DIR / "flow_methods_comparison.png"
    )

    times = [6.68, 1.97, 0.045, 0.19]  

    plot_flow_metrics_cvpr(
        methods=methods,
        msen=msen,
        pepn=pepn,
        times=times,
        title="Optical Flow Performance on Data Stereo Flow Seq. 45",
        save_path=SAVE_DIR / "flow_metrics_cvpr.png"
    )
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D


def plot_assa_vs_deta_hota(
    raw,
    title="HOTA Decomposition: Detection vs Association",
    out_path=None,
    percent=True,
    top_k=6,               # label only top-k points (set 0 to disable labels)
    contour_step=5,        # spacing between contour lines (%)
    pad=3.0,               # axis padding in % units (ignored if percent=False)
    best_line_to="min",    # "min" (to axes mins) or "zero" (to 0)
):
    deta = np.asarray(raw["DetA"], dtype=float)
    assa = np.asarray(raw["AssA"], dtype=float)
    hota = np.asarray(raw["HOTA"], dtype=float)

    scale = 100.0 if percent else 1.0
    deta_s, assa_s, hota_s = deta * scale, assa * scale, hota * scale

    # ---- Axes ranges
    pad_val = pad if percent else pad / 100.0
    x_min = max(0.0, deta_s.min() - pad_val)
    x_max = min(scale, deta_s.max() + pad_val)
    y_min = max(0.0, assa_s.min() - pad_val)
    y_max = min(scale, assa_s.max() + pad_val)

    # ---- Background contours: HOTA = sqrt(DetA * AssA)
    x = np.linspace(x_min, x_max, 250)
    y = np.linspace(y_min, y_max, 250)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(np.maximum(0.0, X) * np.maximum(0.0, Y))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Contour levels with nice rounding
    z_min = np.floor(Z.min() / contour_step) * contour_step
    z_max = np.ceil(Z.max() / contour_step) * contour_step
    levels = np.arange(z_min, z_max + contour_step, contour_step)

    cs = ax.contour(X, Y, Z, levels=levels, linewidths=1.0, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.0f")

    # ---- Scatter colored by HOTA
    sc = ax.scatter(
        deta_s, assa_s,
        c=hota_s,
        cmap="viridis",
        s=65,
        edgecolors="black",
        linewidths=0.4,
        alpha=0.95,
        zorder=3
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("HOTA" + (" (%)" if percent else ""))

    # ---- Best point + dashed projection lines
    order = np.argsort(-hota_s)
    best = order[0]

    # emphasize best as a normal filled dot (no white circle)
    ax.scatter(
        [deta_s[best]], [assa_s[best]],
        c=[hota_s[best]],
        cmap="viridis",
        s=120,                # bigger than others
        edgecolors="black",   # stronger outline
        linewidths=1.2,
        zorder=4
    )

    # dashed lines: to axis mins (recommended) or to 0
    x0 = 0.0 if best_line_to == "zero" else x_min
    y0 = 0.0 if best_line_to == "zero" else y_min
    ax.plot([deta_s[best], deta_s[best]], [y0, assa_s[best]], linestyle="--", linewidth=1.2, zorder=2)
    ax.plot([x0, deta_s[best]], [assa_s[best], assa_s[best]], linestyle="--", linewidth=1.2, zorder=2)

    # ---- Label only top-k ranks (optional)
    if top_k and top_k > 0:
        k = min(top_k, len(order))
        for rank, idx in enumerate(order[:k], start=1):
            ax.text(
                deta_s[idx], assa_s[idx], str(rank),
                fontsize=10,
                ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                zorder=5,
                clip_on=True
            )

    ax.set_xlabel("DetA" + (" (%)" if percent else ""))
    ax.set_ylabel("AssA" + (" (%)" if percent else ""))
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)
    else:
        plt.show()


def create_short_label(run):
    m = run.get("metadata")
    det = m.get("detection_model")
    trk = m.get("tracking_model")
    match = m.get("matching")
    min_confidence = m.get("min_confidence")
    min_iou = m.get("min_iou")
    max_age = m.get("max_age")
    bits = [
        f"{match}",
        f"{max_age}"
    ]

    return "_".join(map(str, bits))


def plot_hota_vs_alpha(runs, title="HOTA vs alpha", out_path=None):
    """
    runs: list of dicts like:
      {"label": "run_name", "raw": raw_dict, "summary": summary_dict}
    """
    plt.figure()
    for r in runs:
        alpha = np.asarray(r["raw"]["alpha"], dtype=float)
        hota = np.asarray(r["raw"]["HOTA"], dtype=float)
        lbl = create_short_label(r)
        mean_h = float(np.mean(hota))
        plt.plot(alpha, hota, label=f"{lbl} ({mean_h:.3f})")

    plt.xlabel("alpha")
    plt.ylabel("HOTA")
    plt.title(title)
    plt.ylim(0, 1)


    plt.legend(
        loc="lower left",
        fontsize=8.5,
        handlelength=1.5,
        borderpad=0.4,
        labelspacing=0.3
    )

    # Format explanation box (top right inside plot)
    format_text = (
        "Label format:\n"
        "matcher_age\n"
    )

    plt.gca().text(
        0.98, 0.98,
        format_text,
        transform=plt.gca().transAxes,
        fontsize=8.5,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()


def plot_hota0_vs_loca0(runs, title="HOTA(0) vs Localization Accuracy", out_path=None, percent=True):
    scale = 100.0 if percent else 1.0

    xs, ys, scores = [], [], []
    for r in runs:
        s = r["summary"]
        x = float(s.get("LocA_0")) * scale
        y = float(s.get("HOTA_0")) * scale
        score = float(s.get("HOTALocA_0")) * scale
        xs.append(x); ys.append(y); scores.append(score)

    xs = np.array(xs); ys = np.array(ys); scores = np.array(scores)

    # Contours for HOTALocA(0) = HOTA(0) * LocA(0) (matching your metric)
    xg = np.linspace(np.nanmin(xs) - 5, np.nanmax(xs) + 5, 200)
    yg = np.linspace(np.nanmin(ys) - 5, np.nanmax(ys) + 5, 200)
    X, Y = np.meshgrid(xg, yg)
    Z = (X * Y) / scale  # because both X and Y are scaled; keep contour scale roughly consistent

    plt.figure()
    cs = plt.contour(X, Y, Z, levels=10)
    plt.clabel(cs, inline=True, fontsize=8)

    plt.scatter(xs, ys)

    # Label order: best HOTALocA_0 rank 1
    order = np.argsort(-scores)
    for rank, idx in enumerate(order, start=1):
        plt.text(xs[idx], ys[idx], str(rank))

    best = order[0]
    plt.plot([xs[best], xs[best]], [yg.min(), ys[best]], linestyle="--")
    plt.plot([xg.min(), xs[best]], [ys[best], ys[best]], linestyle="--")

    plt.xlabel("LocA(0)" + (" (%)" if percent else ""))
    plt.ylabel("HOTA(0)" + (" (%)" if percent else ""))
    plt.title(title)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def plot_idp_vs_idr(
    runs,
    title="IDP vs IDR (IDF1)",
    out_path=None,
    levels=10,
    show_all=False,
    legend_loc="upper left",
    score="idf1",
):
    """
    IDP vs IDR with IDF1 iso-contours + colorbar (IDF1).

    - Each plotted run colored by its IDF1 value.
    - Colorbar on the right shows IDF1 scale.
    - If show_all=False: plots BEST and WORST per (tracking_model, matching).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def meta(run):
        return run.get("metadata", {}) or {}

    def group_key(run):
        m = meta(run)
        trk = str(m.get("tracking_model", "na")).lower()
        match = str(m.get("matching", "na")).lower()
        if "hung" in match:
            match = "hungarian"
        elif "greedy" in match:
            match = "greedy"
        return (trk, match)

    def legend_label(run):
        m = meta(run)
        trk = m.get("tracking_model", "na")
        match = m.get("matching", "na")
        age = m.get("max_age", "na")
        miou = m.get("min_iou", "na")
        conf = m.get("min_confidence", "na")
        return f"{trk} + {match} | age={age}, iou={miou}, conf={conf}"

    # pull arrays
    idp = np.array([float(r["summary"]["IDP"]) for r in runs], dtype=float)
    idr = np.array([float(r["summary"]["IDR"]) for r in runs], dtype=float)
    idf1 = np.array([float(r["summary"]["IDF1"]) for r in runs], dtype=float)

    # choose indices to plot
    if show_all:
        idx_to_plot = list(range(len(runs)))
    else:
        groups = {}
        for i, r in enumerate(runs):
            groups.setdefault(group_key(r), []).append(i)

        best_idx = {}
        worst_idx = {}
        for gk, idxs in groups.items():
            best_idx[gk] = max(idxs, key=lambda i: idf1[i])
            worst_idx[gk] = min(idxs, key=lambda i: idf1[i])

        idx_to_plot = sorted(set(list(best_idx.values()) + list(worst_idx.values())))

    # contour grid
    x = np.linspace(0, 1, 300)
    y = np.linspace(0, 1, 300)
    X, Y = np.meshgrid(x, y)
    Z = (2 * X * Y) / np.maximum(1e-12, (X + Y))

    fig, ax = plt.subplots()

    cs = ax.contour(X, Y, Z, levels=levels, linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=8)

    # --- scatter colored by IDF1 ---
    sc = ax.scatter(
        idr[idx_to_plot],
        idp[idx_to_plot],
        c=idf1[idx_to_plot],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=1.0,
        zorder=3,
    )

    # colorbar
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("IDF1")

    # legend (still one entry per plotted run)
    legend_items = [
        Line2D([0], [0], marker='o', linestyle='', markersize=8,
               markerfacecolor=sc.cmap(sc.norm(idf1[idx])),
               markeredgecolor="black",
               label=legend_label(runs[idx]))
        for idx in idx_to_plot
    ]

    ax.legend(handles=legend_items, loc=legend_loc, fontsize=8, framealpha=0.9)

    ax.set_xlabel("IDR")
    ax.set_ylabel("IDP")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()

def plot_hota_vs_idf1(runs, title="HOTA vs IDF1", out_path=None,
                      zoom_margin_x=0.01, zoom_margin_y=0.01,
                      legend_loc="upper left"):
    """
    Plots HOTA vs IDF1 for the BEST run within each of the 4 combos:
      - Kalman(SORT) × {Hungarian, Greedy}
      - MaxOverlap   × {Hungarian, Greedy}

    Changes vs your previous version:
      - Zoomed axes to data range (+ margins)
      - Dashed guide lines now intersect the *zoomed* axes
      - Best point is highlighted by a thicker edge + larger marker (no hollow overlay)
      - Legend does NOT include a "Best" entry and can be moved to avoid overlap
    """

    def combo_key(run):
        m = run.get("metadata", {}) or {}

        # Track model (robust normalization)
        trk_raw = str(m.get("tracking_model", "")).lower()
        det_raw = str(m.get("detection_model", "")).lower()  # optional, not used
        if any(s in trk_raw for s in ["sort", "kalman"]):
            tracker = "Kalman (SORT)"
        elif any(s in trk_raw for s in ["overlap", "iou", "max_overlap"]):
            tracker = "Max Overlap"
        else:
            # fallback: if you used a tracker_name field or label encodes it
            tracker = "Max Overlap"  # or "Unknown"

        # Matcher (robust)
        match_raw = str(m.get("matching", "")).lower()
        if "hung" in match_raw:
            matcher = "Hungarian"
        elif "greedy" in match_raw:
            matcher = "Greedy"
        else:
            matcher = "Greedy"  # fallback

        return (tracker, matcher)

    categories = [
        ("Kalman (SORT)", "Hungarian"),
        ("Kalman (SORT)", "Greedy"),
        ("Max Overlap", "Hungarian"),
        ("Max Overlap", "Greedy"),
    ]

    # 4 distinct colors (matplotlib default cycle: C0..C3)
    color_map = {
        categories[0]: "C0",
        categories[1]: "C1",
        categories[2]: "C2",
        categories[3]: "C3",
    }

    # -------- extract arrays
    hota = np.array([float(r["summary"]["HOTA"]) for r in runs], dtype=float)
    idf1 = np.array([float(r["summary"]["IDF1"]) for r in runs], dtype=float)
    keys = [combo_key(r) for r in runs]
    print

    # -------- pick BEST run per combo (by HOTA+IDF1; switch to just HOTA if you prefer)
    best_idx_per_cat = {}
    for i, cat in enumerate(keys):
        if cat not in best_idx_per_cat:
            best_idx_per_cat[cat] = i
        else:
            j = best_idx_per_cat[cat]
            if (hota[i] + idf1[i]) > (hota[j] + idf1[j]):
                best_idx_per_cat[cat] = i

    sel = [best_idx_per_cat[cat] for cat in categories if cat in best_idx_per_cat]
    sel_hota = hota[sel]
    sel_idf1 = idf1[sel]
    sel_keys = [keys[i] for i in sel]

    fig, ax = plt.subplots()

    # -------- plot the 4 points (best per combo)
    for x, y, cat in zip(sel_hota, sel_idf1, sel_keys):
        ax.scatter(
            x, y,
            s=130,
            color=color_map[cat],
            edgecolors="black",
            linewidths=1.2,
            zorder=3
        )

    # -------- define zoom window
    x_min = float(sel_hota.min() - zoom_margin_x)
    x_max = float(sel_hota.max() + zoom_margin_x)
    y_min = float(sel_idf1.min() - zoom_margin_y)
    y_max = float(sel_idf1.max() + zoom_margin_y)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # -------- best overall (top-right) among the selected points
    best_local = int(np.argmax(sel_hota + sel_idf1))
    bx, by = float(sel_hota[best_local]), float(sel_idf1[best_local])
    best_cat = sel_keys[best_local]

    # highlight best point: larger + thicker edge (NO hollow overlay)
    ax.scatter(
        bx, by,
        s=190,
        color=color_map[best_cat],
        edgecolors="black",
        linewidths=2.8,
        zorder=5
    )

    # -------- dashed guides that intersect the *zoomed* axes
    ax.plot([bx, bx], [y_min, by], linestyle="--", linewidth=1.2)
    ax.plot([x_min, bx], [by, by], linestyle="--", linewidth=1.2)

    # label at the point
    x_range = x_max - x_min
    y_range = y_max - y_min

    # If point is near right edge → place text to the left
    if bx > x_min + 0.8 * x_range:
        text_x = bx - 0.02 * x_range
        ha = "right"
    else:
        text_x = bx + 0.02 * x_range
        ha = "left"

    ax.text(
        text_x,
        by + 0.01 * y_range,
        f"({bx:.3f}, {by:.3f})",
        fontsize=10,
        ha=ha,
        va="bottom"
    )

    # -------- legend (4 combos only, no "best" entry)
    legend_items = [
        Line2D([0], [0], marker='o', linestyle='', markersize=9,
               markerfacecolor=color_map[cat], markeredgecolor="black",
               label=f"{cat[0]} + {cat[1]}")
        for cat in categories
        if cat in best_idx_per_cat
    ]
    ax.legend(handles=legend_items, loc=legend_loc, fontsize=10, framealpha=0.9)

    ax.set_xlabel("HOTA (mean over α)")
    ax.set_ylabel("IDF1")
    ax.set_title(title)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close()
    else:
        plt.show()
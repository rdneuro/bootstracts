# -*- coding: utf-8 -*-
"""
bootstracts.viz
==================================================

Publication-quality visualization for probabilistic connectomics.

Provides figures for SC uncertainty maps, edge reliability, community
detection results, graph metric distributions, and clinical correlation
plots.  All plots follow journal-ready defaults (300 dpi, tight layout,
colorblind-safe palettes).

Functions
---------
plot_sc_uncertainty
    SC mean, std, and CV matrices with edge-level confidence intervals.
plot_edge_classification
    Robust / present / fragile / spurious classification map.
plot_community_results
    Co-assignment matrix, consensus partition, and node stability.
plot_graph_metrics_ci
    Global graph metrics with bootstrap confidence intervals.
plot_stability_vs_clinical
    Node stability vs clinical variables (e.g., MoCA, ventilation days).

References
----------
- Tournier et al. (2019). NeuroImage 202:116137. MRtrix3.
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm, patches
from typing import Optional, Dict, Tuple, List

from .core import BootstrapResult


# =============================================================================
# STYLE
# =============================================================================

def _setup_style():
    """Apply publication-ready matplotlib defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# =============================================================================
# COLOR UTILITIES
# =============================================================================

_EDGE_CLASS_CMAP = colors.ListedColormap([
    "#f0f0f0",   # 0 = absent (light gray)
    "#e74c3c",   # 1 = spurious (red)
    "#f39c12",   # 2 = fragile (orange)
    "#3498db",   # 3 = present (blue)
    "#2ecc71",   # 4 = robust (green)
])
_EDGE_CLASS_BOUNDS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
_EDGE_CLASS_NORM = colors.BoundaryNorm(_EDGE_CLASS_BOUNDS, _EDGE_CLASS_CMAP.N)
_EDGE_CLASS_LABELS = ["Absent", "Spurious", "Fragile", "Present", "Robust"]


# =============================================================================
# SC UNCERTAINTY
# =============================================================================

def plot_sc_uncertainty(
    result: BootstrapResult,
    parcel_labels: Optional[List[str]] = None,
    log_scale: bool = True,
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    SC mean, standard deviation, and coefficient of variation.

    Three-panel figure showing the bootstrap-averaged SC matrix,
    its standard deviation (uncertainty), and the coefficient of
    variation (relative uncertainty).  Optionally uses log scale
    for better visibility of weak connections.

    Parameters
    ----------
    result : BootstrapResult
    parcel_labels : list of str, optional
        Region labels for tick marks.  Omitted if None.
    log_scale : bool
        If True, display SC mean and std on log₁₀ scale.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    panels = [
        (result.sc_mean, "SC Mean", "viridis"),
        (result.sc_std, "SC Std", "magma"),
        (result.sc_cv, "Coefficient of Variation", "inferno"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        plot_data = data.copy()

        if log_scale and title != "Coefficient of Variation":
            plot_data = np.log10(plot_data + 1e-10)
            vmin = np.percentile(plot_data[plot_data > -9], 5)
            vmax = np.percentile(plot_data[plot_data > -9], 95)
            label = f"log₁₀({title.split()[-1].lower()})"
        else:
            vmin = 0
            vmax = np.percentile(data[data > 0], 95) if np.any(data > 0) else 1
            label = title.split()[-1].lower() if title != "Coefficient of Variation" else "CV"

        im = ax.imshow(plot_data, cmap=cmap, vmin=vmin, vmax=vmax,
                        aspect="equal", interpolation="none")
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        ax.set_title(title, fontweight="bold")

        if parcel_labels is not None and len(parcel_labels) <= 30:
            ax.set_xticks(range(len(parcel_labels)))
            ax.set_xticklabels(parcel_labels, rotation=90, fontsize=6)
            ax.set_yticks(range(len(parcel_labels)))
            ax.set_yticklabels(parcel_labels, fontsize=6)
        else:
            ax.set_xlabel("Region")
            ax.set_ylabel("Region")

    n_parcels = result.n_parcels
    triu = np.triu_indices(n_parcels, k=1)
    mean_cv = np.mean(result.sc_cv[triu][result.sc_mean[triu] > 0])
    fig.suptitle(
        f"SC Uncertainty  |  {result.n_bootstrap} bootstraps  ·  "
        f"{result.n_streamlines:,} streamlines  ·  mean CV = {mean_cv:.3f}",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# EDGE CLASSIFICATION
# =============================================================================

def plot_edge_classification(
    edge_classification: Dict,
    result: Optional[BootstrapResult] = None,
    parcel_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Edge reliability classification map and summary pie chart.

    Left panel: matrix colored by edge class (robust, present, fragile,
    spurious).  Right panel: pie chart with edge counts per category.

    Parameters
    ----------
    edge_classification : dict
        Output from ``classify_edges()``.
    result : BootstrapResult, optional
        If provided, adds bootstrap info to the title.
    parcel_labels : list of str, optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    labels_mat = edge_classification["labels"]
    N = labels_mat.shape[0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # --- Left: Classification matrix ---
    im = ax1.imshow(
        labels_mat, cmap=_EDGE_CLASS_CMAP, norm=_EDGE_CLASS_NORM,
        aspect="equal", interpolation="none",
    )

    # Legend patches
    legend_patches = [
        patches.Patch(color=_EDGE_CLASS_CMAP(i / 4), label=lbl)
        for i, lbl in enumerate(_EDGE_CLASS_LABELS)
        if i > 0  # skip absent
    ]
    ax1.legend(
        handles=legend_patches, loc="upper right",
        fontsize=8, framealpha=0.9,
    )

    if parcel_labels is not None and len(parcel_labels) <= 30:
        ax1.set_xticks(range(len(parcel_labels)))
        ax1.set_xticklabels(parcel_labels, rotation=90, fontsize=6)
        ax1.set_yticks(range(len(parcel_labels)))
        ax1.set_yticklabels(parcel_labels, fontsize=6)

    ax1.set_title("Edge Reliability Classification", fontweight="bold")

    # --- Right: Pie chart ---
    counts = [
        edge_classification["n_robust"],
        edge_classification["n_present"],
        edge_classification["n_fragile"],
        edge_classification["n_spurious"],
    ]
    pie_labels = ["Robust", "Present", "Fragile", "Spurious"]
    pie_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    # Filter out zero categories
    nonzero = [(c, l, col) for c, l, col in zip(counts, pie_labels, pie_colors) if c > 0]
    if nonzero:
        c_nz, l_nz, col_nz = zip(*nonzero)
        ax2.pie(
            c_nz, labels=l_nz, colors=col_nz,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9},
        )
    ax2.set_title("Edge Distribution", fontweight="bold")

    total = sum(counts)
    n_possible = N * (N - 1) // 2
    suptitle = f"Edge Classification  |  {total:,} / {n_possible:,} edges detected"
    if result is not None:
        suptitle += f"  ·  {result.n_bootstrap} bootstraps"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# COMMUNITY RESULTS
# =============================================================================

def plot_community_results(
    community_results: Dict,
    parcel_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Co-assignment matrix, consensus partition, and node stability.

    Three-panel figure:
        1. Co-assignment probability matrix (sorted by consensus partition).
        2. Number of communities across bootstrap runs (histogram).
        3. Node stability distribution and per-node barplot.

    Parameters
    ----------
    community_results : dict
        Output from ``probabilistic_community_detection()``.
    parcel_labels : list of str, optional
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    coassign = community_results["coassignment"]
    consensus = community_results["consensus_partition"]
    stability = community_results["node_stability"]
    n_comm_dist = community_results["n_communities_distribution"]
    mod_dist = community_results["modularity_distribution"]
    N = len(consensus)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --- Panel 1: Co-assignment matrix (sorted by consensus) ---
    sort_idx = np.argsort(consensus)
    sorted_coassign = coassign[np.ix_(sort_idx, sort_idx)]

    ax1 = axes[0]
    im = ax1.imshow(
        sorted_coassign, cmap="YlOrRd", vmin=0, vmax=1,
        aspect="equal", interpolation="none",
    )
    fig.colorbar(im, ax=ax1, label="P(same community)", shrink=0.8)

    # Draw community boundaries
    sorted_consensus = consensus[sort_idx]
    boundaries = np.where(np.diff(sorted_consensus) != 0)[0] + 0.5
    for b in boundaries:
        ax1.axhline(b, color="black", linewidth=0.5, alpha=0.7)
        ax1.axvline(b, color="black", linewidth=0.5, alpha=0.7)

    ax1.set_title("Co-assignment Probability", fontweight="bold")
    ax1.set_xlabel("Region (sorted)")
    ax1.set_ylabel("Region (sorted)")

    # --- Panel 2: N communities histogram ---
    ax2 = axes[1]
    unique_n, counts_n = np.unique(n_comm_dist, return_counts=True)
    ax2.bar(unique_n, counts_n, color="#3498db", edgecolor="white")
    ax2.axvline(
        np.median(n_comm_dist), color="#e74c3c",
        linestyle="--", linewidth=2, label=f"Median = {np.median(n_comm_dist):.0f}",
    )
    ax2.set_xlabel("Number of Communities")
    ax2.set_ylabel("Count")
    ax2.set_title("Community Count Distribution", fontweight="bold")
    ax2.legend(fontsize=9)

    # Add modularity info
    ax2.text(
        0.95, 0.85,
        f"Q = {mod_dist.mean():.3f} ± {mod_dist.std():.3f}",
        transform=ax2.transAxes, ha="right", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # --- Panel 3: Node stability ---
    ax3 = axes[2]
    sort_stab = np.argsort(stability)[::-1]
    bar_colors = plt.cm.RdYlGn(stability[sort_stab])
    ax3.bar(range(N), stability[sort_stab], color=bar_colors, width=1.0)
    ax3.axhline(
        stability.mean(), color="black",
        linestyle="--", linewidth=1.5,
        label=f"Mean = {stability.mean():.3f}",
    )
    ax3.set_xlabel("Region (sorted by stability)")
    ax3.set_ylabel("Stability")
    ax3.set_title("Node Stability", fontweight="bold")
    ax3.set_xlim(-0.5, N - 0.5)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=9)

    n_cons = len(np.unique(consensus))
    fig.suptitle(
        f"Probabilistic Community Detection  |  "
        f"{n_cons} consensus communities  ·  "
        f"mean stability = {stability.mean():.3f}",
        fontsize=11, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# GRAPH METRICS WITH CI
# =============================================================================

def plot_graph_metrics_ci(
    metrics: Dict,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Global graph metrics with bootstrap confidence intervals.

    Violin + strip plots for each global metric (density, mean strength,
    modularity, global efficiency, transitivity), showing the full
    bootstrap distribution and 95% CI.

    Parameters
    ----------
    metrics : dict
        Output from ``graph_metrics_with_ci()``.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    # Select global metrics (those with scalar 'mean')
    global_names = [
        k for k, v in metrics.items()
        if isinstance(v, dict) and "values" in v and np.isscalar(v.get("mean", None))
    ]

    n_metrics = len(global_names)
    if n_metrics == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No global metrics found", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(
        2, (n_metrics + 1) // 2, figsize=figsize, squeeze=False,
    )
    axes_flat = axes.ravel()

    palette = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for idx, name in enumerate(global_names):
        ax = axes_flat[idx]
        m = metrics[name]
        values = m["values"]
        ci = m["ci"]
        mean = m["mean"]

        # Violin plot
        parts = ax.violinplot(
            values, positions=[0], showmeans=False, showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(palette[idx])
            pc.set_alpha(0.6)

        # CI bar
        ax.plot([0, 0], [ci[0], ci[1]], color="black", linewidth=3)
        ax.plot(0, mean, "o", color="black", markersize=8, zorder=5)

        # Scatter (jittered)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(values))
        ax.scatter(
            jitter, values, s=2, alpha=0.15, color=palette[idx],
            rasterized=True,
        )

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(name.replace("_", " ").title(), fontweight="bold")

        # Annotate
        ax.text(
            0.95, 0.05,
            f"{mean:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Hide extra axes
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Graph Metrics with Bootstrap 95% CI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# CLINICAL CORRELATION
# =============================================================================

def plot_stability_vs_clinical(
    node_stability: np.ndarray,
    clinical_variable: np.ndarray,
    clinical_name: str = "Clinical Score",
    parcel_labels: Optional[List[str]] = None,
    highlight_top_n: int = 5,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Node stability (or any nodal metric) vs a clinical variable.

    Designed for per-region analysis across subjects.  If used with
    single-subject node stability, plots stability vs region index
    colored by the clinical variable.  If used with group-level data
    (one value per subject), plots the correlation.

    Left panel: scatter plot with regression line and Pearson r.
    Right panel: regional barplot highlighting top-N and bottom-N regions.

    Parameters
    ----------
    node_stability : np.ndarray
        Per-node metric (e.g., from community detection).
    clinical_variable : np.ndarray
        Clinical scores (same length as node_stability for per-region,
        or scalar repeated for each node).
    clinical_name : str
        Label for the clinical variable axis.
    parcel_labels : list of str, optional
    highlight_top_n : int
        Number of top/bottom regions to label.
    figsize : tuple
    save_path : str, optional

    Returns
    -------
    matplotlib.Figure
    """
    _setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    N = len(node_stability)
    valid = ~(np.isnan(node_stability) | np.isnan(clinical_variable))
    x = node_stability[valid]
    y = clinical_variable[valid]

    # --- Left: Scatter + regression ---
    ax1.scatter(x, y, s=30, alpha=0.6, color="#3498db", edgecolors="white",
                linewidths=0.5)

    if len(x) > 2:
        from scipy.stats import pearsonr
        r, p = pearsonr(x, y)
        coeffs = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax1.plot(x_line, np.polyval(coeffs, x_line), "r-", linewidth=2,
                 label=f"r = {r:.3f}, p = {p:.4f}")
        ax1.legend(fontsize=10, loc="best")

    ax1.set_xlabel("Node Stability")
    ax1.set_ylabel(clinical_name)
    ax1.set_title(f"Stability vs {clinical_name}", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # --- Right: Regional barplot ---
    sort_idx = np.argsort(node_stability)[::-1]
    bar_colors = np.full(N, "#bdc3c7")

    # Top-N in green, bottom-N in red
    for i in range(min(highlight_top_n, N)):
        bar_colors[sort_idx[i]] = "#2ecc71"
    for i in range(min(highlight_top_n, N)):
        bar_colors[sort_idx[-(i + 1)]] = "#e74c3c"

    ax2.bar(range(N), node_stability[sort_idx],
            color=[bar_colors[i] for i in sort_idx], width=1.0)

    # Label top/bottom regions
    if parcel_labels is not None:
        for i in range(min(highlight_top_n, N)):
            idx = sort_idx[i]
            label = parcel_labels[idx] if idx < len(parcel_labels) else f"R{idx}"
            ax2.text(i, node_stability[idx] + 0.01, label,
                     rotation=45, fontsize=6, ha="left")

    ax2.set_xlabel("Region (sorted)")
    ax2.set_ylabel("Node Stability")
    ax2.set_title("Regional Stability Ranking", fontweight="bold")
    ax2.set_xlim(-0.5, N - 0.5)

    # Legend
    legend_elements = [
        patches.Patch(facecolor="#2ecc71", label=f"Top {highlight_top_n}"),
        patches.Patch(facecolor="#e74c3c", label=f"Bottom {highlight_top_n}"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

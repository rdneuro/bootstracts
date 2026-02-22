# -*- coding: utf-8 -*-
"""
bootstracts.viz_extended
==================================================

Extended visualization for advanced analyses.

Adds publication-quality figures for:
    - NBS results (significant components)
    - Rich club curves with CIs
    - Hub classification maps
    - Communicability matrices
    - Along-tract profiles with CIs
    - CPM brain-behavior predictions
    - PLS latent variables

All plots follow journal-ready defaults (300 dpi, tight layout,
colorblind-safe palettes).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from typing import Optional, Dict, List, Tuple


def _setup_style():
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
# NBS RESULTS
# =============================================================================

def plot_nbs_results(
    nbs_results: Dict,
    parcel_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    NBS significant components and t-statistic map.

    Three panels:
        1. Edge-wise t-statistic matrix
        2. Significant edges binary map
        3. Component summary (sizes + p-values)
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    t_stats = nbs_results["t_stats"]
    sig_edges = nbs_results["significant_edges"]
    N = t_stats.shape[0]

    # Panel 1: t-statistic map
    ax = axes[0]
    vmax = np.percentile(np.abs(t_stats[t_stats != 0]), 95) if (t_stats != 0).any() else 1
    im = ax.imshow(
        t_stats, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        aspect="equal", interpolation="none",
    )
    fig.colorbar(im, ax=ax, label="t-statistic", shrink=0.8)
    ax.set_title("Edge-wise t-statistics", fontweight="bold")

    # Panel 2: Significant edges
    ax = axes[1]
    sig_display = sig_edges.astype(float)
    cmap_sig = colors.ListedColormap(["#f0f0f0", "#e74c3c"])
    ax.imshow(sig_display, cmap=cmap_sig, vmin=0, vmax=1,
              aspect="equal", interpolation="none")
    n_sig = int(sig_edges[np.triu_indices(N, k=1)].sum())
    ax.set_title(
        f"Significant edges (n={n_sig})", fontweight="bold"
    )

    # Panel 3: Component summary
    ax = axes[2]
    comp_sizes = nbs_results["component_sizes"]
    p_vals = nbs_results["p_values"]

    if comp_sizes:
        bar_colors = [
            "#2ecc71" if p < 0.05 else "#bdc3c7"
            for p in p_vals
        ]
        bars = ax.bar(range(len(comp_sizes)), comp_sizes, color=bar_colors)
        ax.set_xlabel("Component")
        ax.set_ylabel("Number of edges")
        ax.set_title("Component sizes", fontweight="bold")

        for i, (s, p) in enumerate(zip(comp_sizes, p_vals)):
            ax.text(i, s + 0.5, f"p={p:.3f}", ha="center", fontsize=8)

        legend_elements = [
            patches.Patch(facecolor="#2ecc71", label="p < 0.05"),
            patches.Patch(facecolor="#bdc3c7", label="p ≥ 0.05"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
    else:
        ax.text(
            0.5, 0.5, "No components found",
            ha="center", va="center", transform=ax.transAxes,
        )

    fig.suptitle(
        f"Network-Based Statistic  |  threshold = {nbs_results['threshold']:.1f}  ·  "
        f"{nbs_results['n_permutations']} permutations",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# RICH CLUB
# =============================================================================

def plot_rich_club(
    rich_club_results: Dict,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Rich club Φ(k) and Φ_norm(k) with bootstrap CIs."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    k = rich_club_results["k_values"]

    # Raw Φ(k)
    ax1.fill_between(
        k, rich_club_results["phi_ci_low"],
        rich_club_results["phi_ci_high"],
        alpha=0.3, color="#3498db",
    )
    ax1.plot(k, rich_club_results["phi_mean"], "-o",
             color="#2c3e50", markersize=3, label="Φ(k)")
    ax1.set_xlabel("Degree threshold k")
    ax1.set_ylabel("Rich-club coefficient Φ(k)")
    ax1.set_title("Rich-Club Coefficient", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Normalized Φ_norm(k)
    ax2.fill_between(
        k, rich_club_results["phi_norm_ci_low"],
        rich_club_results["phi_norm_ci_high"],
        alpha=0.3, color="#e74c3c",
    )
    ax2.plot(k, rich_club_results["phi_norm_mean"], "-o",
             color="#c0392b", markersize=3, label="Φ_norm(k)")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Degree threshold k")
    ax2.set_ylabel("Normalized Φ(k)")
    ax2.set_title("Normalized Rich-Club", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Rich-Club Analysis with Bootstrap 95% CI",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# HUB DETECTION
# =============================================================================

def plot_hub_detection(
    hub_results: Dict,
    parcel_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Hub classification under uncertainty.

    Four panels:
        1. Hub probability per node
        2. Guimerà-Amaral cartography (P vs z)
        3. Betweenness centrality with CIs
        4. Classification stability
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    N = len(hub_results["hub_probability"])
    hub_prob = hub_results["hub_probability"]

    # Panel 1: Hub probability
    ax = axes[0, 0]
    sort_idx = np.argsort(hub_prob)[::-1]
    bar_colors = plt.cm.RdYlGn(hub_prob[sort_idx])
    ax.bar(range(N), hub_prob[sort_idx], color=bar_colors, width=1.0)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1,
               label="P = 0.5")
    ax.set_xlabel("Region (sorted)")
    ax.set_ylabel("P(hub)")
    ax.set_title("Hub Probability", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.5, N - 0.5)

    # Panel 2: Guimerà-Amaral cartography
    ax = axes[0, 1]
    P_mean = hub_results["participation_mean"]
    z_mean = hub_results["within_module_z_mean"]
    class_mode = hub_results["hub_class_mode"]

    class_colors = {0: "#bdc3c7", 1: "#3498db", 2: "#e74c3c", 3: "#f39c12"}
    class_names = hub_results.get("hub_class_labels", {
        0: "non-hub", 1: "provincial", 2: "connector", 3: "kinless"
    })

    for c in np.unique(class_mode):
        mask = class_mode == c
        ax.scatter(
            P_mean[mask], z_mean[mask], s=30, alpha=0.7,
            c=class_colors.get(c, "#bdc3c7"),
            label=class_names.get(c, f"class {c}"),
            edgecolors="white", linewidths=0.5,
        )

    ax.axvline(0.3, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.75, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Participation coefficient P")
    ax.set_ylabel("Within-module degree z")
    ax.set_title("Guimerà-Amaral Cartography", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    # Panel 3: Betweenness centrality
    ax = axes[1, 0]
    betw = hub_results["betweenness_mean"]
    betw_ci_l = hub_results.get("betweenness_ci_low", betw)
    betw_ci_h = hub_results.get("betweenness_ci_high", betw)
    sort_b = np.argsort(betw)[::-1]

    ax.bar(range(N), betw[sort_b], color="#3498db", alpha=0.7, width=1.0)
    ax.errorbar(
        range(N), betw[sort_b],
        yerr=[betw[sort_b] - betw_ci_l[sort_b],
              betw_ci_h[sort_b] - betw[sort_b]],
        fmt="none", ecolor="gray", elinewidth=0.5, capsize=0,
    )
    ax.set_xlabel("Region (sorted)")
    ax.set_ylabel("Betweenness centrality")
    ax.set_title("Betweenness with 95% CI", fontweight="bold")
    ax.set_xlim(-0.5, N - 0.5)

    # Panel 4: Classification stability
    ax = axes[1, 1]
    stability = hub_results["hub_class_stability"]
    sort_s = np.argsort(stability)[::-1]
    colors_s = [class_colors.get(class_mode[i], "#bdc3c7") for i in sort_s]
    ax.bar(range(N), stability[sort_s], color=colors_s, width=1.0)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Region (sorted)")
    ax.set_ylabel("Classification stability")
    ax.set_title("Hub Classification Stability", fontweight="bold")
    ax.set_xlim(-0.5, N - 0.5)

    fig.suptitle(
        "Hub Detection Under Uncertainty",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# ALONG-TRACT PROFILES
# =============================================================================

def plot_along_tract_profile(
    profile_results: Dict,
    tract_name: str = "Tract",
    metric_name: str = "FA",
    comparison: Optional[Dict] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Along-tract profile with bootstrap CI ribbon.

    If comparison results are provided (from compare_tract_profiles),
    highlights significant clusters.
    """
    _setup_style()

    if comparison is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                        gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.7))

    n_pts = len(profile_results["mean_profile"])
    x = np.arange(n_pts)

    # Main profile
    ax1.fill_between(
        x, profile_results["ci_low"], profile_results["ci_high"],
        alpha=0.25, color="#3498db", label="95% CI",
    )
    ax1.fill_between(
        x, profile_results["iqr_low"], profile_results["iqr_high"],
        alpha=0.35, color="#3498db", label="IQR",
    )
    ax1.plot(
        x, profile_results["mean_profile"], "-",
        color="#2c3e50", linewidth=2, label="Mean",
    )

    ax1.set_xlabel("Position along tract")
    ax1.set_ylabel(metric_name)
    ax1.set_title(f"{tract_name} — Along-Tract Profile", fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, n_pts - 1)

    # Comparison panel
    if comparison is not None:
        sig_pts = comparison["significant_points"]
        t_stats = comparison["t_stats"]

        ax2.plot(x, t_stats, color="#2c3e50", linewidth=1)
        ax2.fill_between(
            x, t_stats, 0, where=sig_pts,
            color="#e74c3c", alpha=0.4, label="Significant",
        )
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_xlabel("Position along tract")
        ax2.set_ylabel("t-statistic")
        ax2.set_title("Group Comparison", fontweight="bold", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.set_xlim(0, n_pts - 1)

        # Highlight clusters
        for clust in comparison.get("clusters", []):
            ax2.axvspan(
                clust["start"], clust["end"],
                alpha=0.15, color="#e74c3c",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# CPM RESULTS
# =============================================================================

def plot_cpm_results(
    cpm_results: Dict,
    behavior: np.ndarray,
    behavior_name: str = "Behavior",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    CPM prediction results: scatter + edge stability map.

    Two panels:
        1. Predicted vs actual behavior with prediction intervals
        2. Edge selection frequency heatmap
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    predictions = cpm_results["predictions"]
    pi = cpm_results["prediction_intervals"]
    r = cpm_results["correlation"]
    p = cpm_results["p_value"]

    # Panel 1: Predicted vs actual
    ax1.errorbar(
        behavior, predictions,
        yerr=[predictions - pi[:, 0], pi[:, 1] - predictions],
        fmt="o", markersize=5, alpha=0.6, color="#3498db",
        ecolor="#bdc3c7", elinewidth=0.5,
    )
    lims = [
        min(behavior.min(), predictions.min()),
        max(behavior.max(), predictions.max()),
    ]
    ax1.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="Identity")
    coeffs = np.polyfit(behavior, predictions, 1)
    x_line = np.linspace(lims[0], lims[1], 100)
    ax1.plot(x_line, np.polyval(coeffs, x_line), "r-", linewidth=2,
             label=f"r = {r:.3f}, p = {p:.4f}")
    ax1.set_xlabel(f"Actual {behavior_name}")
    ax1.set_ylabel(f"Predicted {behavior_name}")
    ax1.set_title("CPM Predictions (Bagged)", fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Edge selection frequency
    freq = cpm_results["edge_selection_freq"]
    im = ax2.imshow(
        freq, cmap="YlOrRd", vmin=0, vmax=freq.max(),
        aspect="equal", interpolation="none",
    )
    fig.colorbar(im, ax=ax2, label="Selection frequency", shrink=0.8)
    ax2.set_title("Edge Selection Stability", fontweight="bold")
    ax2.set_xlabel("Region")
    ax2.set_ylabel("Region")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# COMMUNICABILITY
# =============================================================================

def plot_communicability(
    comm_results: Dict,
    parcel_labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Communicability matrix and node-level summaries."""
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: Communicability matrix
    comm = comm_results["comm_mean"]
    im = ax1.imshow(
        np.log10(comm + 1e-10), cmap="inferno",
        aspect="equal", interpolation="none",
    )
    fig.colorbar(im, ax=ax1, label="log₁₀(communicability)", shrink=0.8)
    ax1.set_title("Communicability Matrix", fontweight="bold")

    # Panel 2: Subgraph centrality
    N = len(comm_results["subgraph_centrality_mean"])
    sc_mean = comm_results["subgraph_centrality_mean"]
    sort_idx = np.argsort(sc_mean)[::-1]

    ax2.bar(
        range(N), sc_mean[sort_idx],
        color=plt.cm.inferno(sc_mean[sort_idx] / sc_mean.max()),
        width=1.0,
    )
    ax2.set_xlabel("Region (sorted)")
    ax2.set_ylabel("Subgraph centrality")
    ax2.set_title("Subgraph Centrality", fontweight="bold")
    ax2.set_xlim(-0.5, N - 0.5)

    if parcel_labels is not None and N <= 30:
        ax2.set_xticks(range(N))
        ax2.set_xticklabels(
            [parcel_labels[i] for i in sort_idx],
            rotation=90, fontsize=6,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# PLS RESULTS
# =============================================================================

def plot_pls_results(
    pls_results: Dict,
    n_show: int = 3,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """PLS brain-behavior: singular values, bootstrap ratios, scores."""
    _setup_style()

    n_comp = min(n_show, len(pls_results["singular_values"]))
    fig, axes = plt.subplots(2, n_comp, figsize=figsize, squeeze=False)

    sv = pls_results["singular_values"]
    perm_p = pls_results["perm_p_values"]
    bsr = pls_results["bootstrap_ratios"]
    brain_scores = pls_results["brain_scores"]
    behav_scores = pls_results["behavior_scores"]
    var_exp = pls_results["variance_explained"]

    for c in range(n_comp):
        # Top row: Bootstrap ratios
        ax = axes[0, c]
        bsr_c = bsr[:, c]
        reliable = np.abs(bsr_c) > 2

        bar_colors = np.where(reliable, "#e74c3c", "#bdc3c7")
        ax.bar(range(len(bsr_c)), bsr_c, color=bar_colors, width=1.0)
        ax.axhline(2, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(-2, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(
            f"LV{c + 1}  |  p={perm_p[c]:.3f}  |  "
            f"var={var_exp[c]:.1%}",
            fontweight="bold", fontsize=10,
        )
        ax.set_xlabel("Brain feature")
        ax.set_ylabel("BSR")

        # Bottom row: Brain vs behavior scores
        ax = axes[1, c]
        ax.scatter(
            brain_scores[:, c], behav_scores[:, c],
            s=30, alpha=0.7, color="#3498db",
            edgecolors="white", linewidths=0.5,
        )

        from scipy.stats import pearsonr
        r, p = pearsonr(brain_scores[:, c], behav_scores[:, c])
        ax.set_xlabel("Brain scores")
        ax.set_ylabel("Behavior scores")
        ax.set_title(f"r = {r:.3f}", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "PLS Brain-Behavior Analysis",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig

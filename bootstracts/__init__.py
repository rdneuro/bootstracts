# -*- coding: utf-8 -*-
"""
bootstracts
==============================

Probabilistic connectomics via SIFT2 tractogram bootstrap.

The standard connectomics pipeline treats structural connectivity (SC)
as deterministic: SC_ab = Σ wᵢ for streamlines connecting parcels a↔b.
But this is one realization of a stochastic process (seeding, tracking,
filtering), and the uncertainty is real but universally ignored.

This module quantifies that uncertainty via weighted bootstrap of the
SIFT2 tractogram, enabling:

    1. Confidence intervals on every SC edge
    2. Edge reliability classification (robust vs fragile connections)
    3. Probabilistic community detection (module assignment stability)
    4. Graph metric distributions (not point estimates)
    5. Network-level inference (NBS, TFNBS, permutation testing)
    6. Rich club, small-world, and hub analysis under uncertainty
    7. Communicability with bootstrap CIs
    8. Connectome-predictive modeling with bagging
    9. Along-tract profiling with per-node CIs
    10. Voxel-level bootstrap (wild/residual) for signal uncertainty
    11. HDF5/BIDS-compatible storage

The bootstrap operates on (streamline, weight) pairs:
    For b = 1, …, B:
        Sample N streamlines with replacement from the tractogram
        Assign each sampled streamline its SIFT2 weight
        Build SC⁽ᵇ⁾ by aggregating into the parcellation

    Uncertainty at edge (a,b) = variability of SC⁽ᵇ⁾_{ab} across b

Compatible with any MRtrix3 pipeline:
    tckgen → tcksift2 → tck2connectome -out_assignments

Modules
-------
core
    StreamlineAssignment, EdgeStats, BootstrapResult data structures;
    loading from MRtrix3 files; bootstrap engine; edge classification;
    disparity filter backbone extraction.
backends
    GPU acceleration (NumPy/CuPy/JAX) and Welford streaming stats.
community
    Probabilistic community detection, co-assignment matrices,
    consensus partition, node stability, graph metrics with CIs.
graph_analysis
    Rich club, small-world propensity, hub detection (Guimerà-Amaral),
    communicability, participation coefficient — all with bootstrap CIs.
inference
    NBS, TFNBS, permutation testing (edge-wise and global),
    connectome-predictive modeling with bagging, PLS brain-behavior.
along_tract
    Along-tract profiling with bootstrap CIs, group comparison with
    cluster-based correction, bundle membership stability.
voxel_bootstrap
    Wild bootstrap for DTI, residual bootstrap for CSD, full
    voxel-bootstrap connectome pipeline.
storage
    HDF5 I/O, BIDS-compatible export (BEP017, BEP038).
viz
    Publication-quality figures for SC uncertainty, edge classification,
    communities, graph metrics, clinical correlations.
viz_extended
    Extended visualization for NBS, rich club, hubs, along-tract,
    CPM, communicability, and PLS results.

Pipeline
--------
From MRtrix3 outputs::

    from bootstracts import (
        load_streamline_assignments,
        bootstrap_tractogram,
        classify_edges,
        probabilistic_community_detection,
        graph_metrics_with_ci,
        nbs_bootstrap,
        rich_club_bootstrap,
        hub_detection_bootstrap,
        save_full_analysis,
        export_bids,
    )

    assignments = load_streamline_assignments(
        connectome_csv='connectome.csv',
        weights_csv='sift2_weights.csv',
        assignments_txt='assignments.txt',
        n_parcels=100,
    )
    result = bootstrap_tractogram(assignments, n_bootstrap=1000)
    edge_class = classify_edges(result)
    communities = probabilistic_community_detection(result)
    metrics = graph_metrics_with_ci(result)
    hubs = hub_detection_bootstrap(result, community_results=communities)

    save_full_analysis(
        'results.h5', result,
        edge_classification=edge_class,
        community_results=communities,
        graph_metrics=metrics,
        hub_results=hubs,
    )
    export_bids(result, 'derivatives/', 'COVID001', atlas_name='Schaefer100')

For prototyping without raw tractograms::

    from bootstracts import create_assignments_from_sc
    assignments = create_assignments_from_sc(sc_matrix)

References
----------
- Tournier et al. (2019). NeuroImage 202:116137. MRtrix3.
- Smith et al. (2015). NeuroImage 121:176-185. SIFT2.
- Efron & Tibshirani (1993). An Introduction to the Bootstrap.
  Chapman & Hall.
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069.
- Zalesky, Fornito & Bullmore (2010). NeuroImage 53:1197-1207.
- van den Heuvel & Sporns (2011). J Neurosci 31:15775-86.
- Estrada & Hatano (2008). Phys Rev E 77:036111.
- Shen et al. (2017). Nat Protoc 12:506-518.
- Whitcher et al. (2008). Hum Brain Mapp 29:346-362.
- Maier-Hein et al. (2017). Nat Commun 8:1349.
"""

__version__ = "0.2.0"

# === backends ===
from .backends import (
    get_backend,
    available_backends,
    WelfordAccumulator,
    bootstrap_sc_batch_gpu,
)

# === core ===
from .core import (
    StreamlineAssignment,
    EdgeStats,
    BootstrapResult,
    load_streamline_assignments,
    create_assignments_from_sc,
    build_sc_from_streamlines,
    bootstrap_tractogram,
    classify_edges,
    disparity_filter,
)

# === community ===
from .community import (
    probabilistic_community_detection,
    graph_metrics_with_ci,
)

# === graph_analysis ===
from .graph_analysis import (
    rich_club_bootstrap,
    small_world_propensity_bootstrap,
    hub_detection_bootstrap,
    communicability_bootstrap,
)

# === inference ===
from .inference import (
    nbs_bootstrap,
    tfnbs,
    permutation_test_edges,
    permutation_test_global,
    cpm_bootstrap,
    pls_bootstrap,
)

# === along_tract ===
from .along_tract import (
    along_tract_bootstrap,
    compare_tract_profiles,
    bundle_membership_stability,
    tract_profile_from_streamlines,
)

# === voxel_bootstrap ===
from .voxel_bootstrap import (
    VoxelBootstrapConfig,
    wild_bootstrap_dti,
    residual_bootstrap_csd,
    voxel_bootstrap_connectome,
)

# === storage ===
from .storage import (
    save_bootstrap_result,
    load_bootstrap_result,
    save_full_analysis,
    load_full_analysis,
    export_bids,
)

# === viz ===
from .viz import (
    plot_sc_uncertainty,
    plot_edge_classification,
    plot_community_results,
    plot_graph_metrics_ci,
    plot_stability_vs_clinical,
)

# === viz_extended ===
from .viz_extended import (
    plot_nbs_results,
    plot_rich_club,
    plot_hub_detection,
    plot_along_tract_profile,
    plot_cpm_results,
    plot_communicability,
    plot_pls_results,
)

__all__ = [
    # --- version ---
    "__version__",
    # --- backends ---
    "get_backend",
    "available_backends",
    "WelfordAccumulator",
    "bootstrap_sc_batch_gpu",
    # --- core ---
    "StreamlineAssignment",
    "EdgeStats",
    "BootstrapResult",
    "load_streamline_assignments",
    "create_assignments_from_sc",
    "build_sc_from_streamlines",
    "bootstrap_tractogram",
    "classify_edges",
    "disparity_filter",
    # --- community ---
    "probabilistic_community_detection",
    "graph_metrics_with_ci",
    # --- graph_analysis ---
    "rich_club_bootstrap",
    "small_world_propensity_bootstrap",
    "hub_detection_bootstrap",
    "communicability_bootstrap",
    # --- inference ---
    "nbs_bootstrap",
    "tfnbs",
    "permutation_test_edges",
    "permutation_test_global",
    "cpm_bootstrap",
    "pls_bootstrap",
    # --- along_tract ---
    "along_tract_bootstrap",
    "compare_tract_profiles",
    "bundle_membership_stability",
    "tract_profile_from_streamlines",
    # --- voxel_bootstrap ---
    "VoxelBootstrapConfig",
    "wild_bootstrap_dti",
    "residual_bootstrap_csd",
    "voxel_bootstrap_connectome",
    # --- storage ---
    "save_bootstrap_result",
    "load_bootstrap_result",
    "save_full_analysis",
    "load_full_analysis",
    "export_bids",
    # --- viz ---
    "plot_sc_uncertainty",
    "plot_edge_classification",
    "plot_community_results",
    "plot_graph_metrics_ci",
    "plot_stability_vs_clinical",
    # --- viz_extended ---
    "plot_nbs_results",
    "plot_rich_club",
    "plot_hub_detection",
    "plot_along_tract_profile",
    "plot_cpm_results",
    "plot_communicability",
    "plot_pls_results",
]

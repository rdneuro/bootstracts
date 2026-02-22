# -*- coding: utf-8 -*-
"""
bootstracts.storage
==================================================

Persistent storage for bootstrap results.

Provides HDF5 and (optionally) Zarr backends for efficient storage
of bootstrap distributions, summary statistics, and metadata.
Follows BIDS connectivity derivatives conventions (BEP017, BEP038).

HDF5 structure::

    bootstrap_results.h5
    ├── metadata/
    │   ├── atlas (str)
    │   ├── n_bootstrap (int)
    │   ├── n_streamlines (int)
    │   ├── n_parcels (int)
    │   ├── sift2_weighted (bool)
    │   ├── random_seed (int)
    │   └── software_version (str)
    ├── sc_matrices/
    │   ├── mean (N, N)
    │   ├── std (N, N)
    │   ├── cv (N, N)
    │   ├── ci_low (N, N)
    │   ├── ci_high (N, N)
    │   └── p_nonzero (N, N)
    ├── edge_classification/ (N, N) int8
    ├── graph_metrics/
    │   ├── density (B,)
    │   ├── modularity (B,)
    │   └── ...
    └── community/
        ├── coassignment (N, N)
        ├── consensus_partition (N,)
        ├── node_stability (N,)
        └── all_partitions (B, N)

Functions
---------
save_bootstrap_result
    Save BootstrapResult to HDF5.
load_bootstrap_result
    Load BootstrapResult from HDF5.
save_full_analysis
    Save complete analysis (bootstrap + community + graph metrics).
export_bids
    Export results as BIDS-compliant connectivity derivatives.

References
----------
- BEP017: Relationship and Connectivity Matrices.
- BEP038: Atlas Specification.
"""

import numpy as np
import json
import os
from typing import Dict, Optional, Union
from pathlib import Path

from .core import BootstrapResult


# Package version
__version__ = "0.2.0"


# =============================================================================
# HDF5 I/O
# =============================================================================

def save_bootstrap_result(
    result: BootstrapResult,
    filepath: str,
    compression: str = "gzip",
    compression_opts: int = 4,
    metadata: Optional[Dict] = None,
    include_samples: bool = False,
) -> None:
    """
    Save BootstrapResult to HDF5.

    Parameters
    ----------
    result : BootstrapResult
    filepath : str
        Output HDF5 path.
    compression : str
        HDF5 compression filter.
    compression_opts : int
        Compression level (1-9 for gzip).
    metadata : dict, optional
        Additional metadata (atlas name, subject ID, etc.).
    include_samples : bool
        If True and sc_samples is not None, store all samples.
    """
    import h5py

    with h5py.File(filepath, "w") as f:
        # Metadata
        meta = f.create_group("metadata")
        meta.attrs["n_bootstrap"] = result.n_bootstrap
        meta.attrs["n_streamlines"] = result.n_streamlines
        meta.attrs["n_parcels"] = result.n_parcels
        meta.attrs["software_version"] = __version__

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta.attrs[k] = v

        # SC summary statistics
        sc_grp = f.create_group("sc_matrices")
        kw = dict(compression=compression, compression_opts=compression_opts)

        sc_grp.create_dataset("mean", data=result.sc_mean, **kw)
        sc_grp.create_dataset("std", data=result.sc_std, **kw)
        sc_grp.create_dataset("cv", data=result.sc_cv, **kw)
        sc_grp.create_dataset("ci_low", data=result.sc_ci_low, **kw)
        sc_grp.create_dataset("ci_high", data=result.sc_ci_high, **kw)
        sc_grp.create_dataset("p_nonzero", data=result.prob_nonzero, **kw)

        if result.sc_observed is not None:
            sc_grp.create_dataset("observed", data=result.sc_observed, **kw)

        # Full samples (optional, large)
        if include_samples and result.sc_samples is not None:
            sc_grp.create_dataset(
                "samples", data=result.sc_samples,
                chunks=(1, result.n_parcels, result.n_parcels),
                **kw,
            )


def load_bootstrap_result(filepath: str) -> BootstrapResult:
    """
    Load BootstrapResult from HDF5.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    BootstrapResult
    """
    import h5py

    with h5py.File(filepath, "r") as f:
        meta = f["metadata"]
        sc = f["sc_matrices"]

        sc_samples = None
        if "samples" in sc:
            sc_samples = sc["samples"][:]

        sc_observed = None
        if "observed" in sc:
            sc_observed = sc["observed"][:]

        return BootstrapResult(
            sc_mean=sc["mean"][:],
            sc_std=sc["std"][:],
            sc_cv=sc["cv"][:],
            sc_ci_low=sc["ci_low"][:],
            sc_ci_high=sc["ci_high"][:],
            prob_nonzero=sc["p_nonzero"][:],
            n_bootstrap=int(meta.attrs["n_bootstrap"]),
            n_streamlines=int(meta.attrs["n_streamlines"]),
            n_parcels=int(meta.attrs["n_parcels"]),
            sc_samples=sc_samples,
            sc_observed=sc_observed,
        )


def save_full_analysis(
    filepath: str,
    result: BootstrapResult,
    edge_classification: Optional[Dict] = None,
    community_results: Optional[Dict] = None,
    graph_metrics: Optional[Dict] = None,
    hub_results: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    compression: str = "gzip",
) -> None:
    """
    Save complete analysis to HDF5.

    Parameters
    ----------
    filepath : str
    result : BootstrapResult
    edge_classification : dict, optional
        Output from classify_edges.
    community_results : dict, optional
        Output from probabilistic_community_detection.
    graph_metrics : dict, optional
        Output from graph_metrics_with_ci.
    hub_results : dict, optional
        Output from hub_detection_bootstrap.
    metadata : dict, optional
    compression : str
    """
    import h5py

    kw = dict(compression=compression)

    # First save the bootstrap result
    save_bootstrap_result(
        result, filepath, metadata=metadata, compression=compression
    )

    with h5py.File(filepath, "a") as f:
        # Edge classification
        if edge_classification is not None:
            ec_grp = f.create_group("edge_classification")
            ec_grp.create_dataset(
                "labels",
                data=edge_classification["labels"].astype(np.int8),
                **kw,
            )
            for k in ["n_robust", "n_present", "n_fragile", "n_spurious"]:
                ec_grp.attrs[k] = edge_classification[k]

        # Community detection
        if community_results is not None:
            com_grp = f.create_group("community")
            com_grp.create_dataset(
                "coassignment",
                data=community_results["coassignment"],
                **kw,
            )
            com_grp.create_dataset(
                "consensus_partition",
                data=community_results["consensus_partition"],
            )
            com_grp.create_dataset(
                "node_stability",
                data=community_results["node_stability"],
            )
            if "all_partitions" in community_results:
                com_grp.create_dataset(
                    "all_partitions",
                    data=community_results["all_partitions"],
                    **kw,
                )
            com_grp.create_dataset(
                "n_communities_distribution",
                data=community_results["n_communities_distribution"],
            )
            com_grp.create_dataset(
                "modularity_distribution",
                data=community_results["modularity_distribution"],
            )

        # Graph metrics
        if graph_metrics is not None:
            gm_grp = f.create_group("graph_metrics")
            for name, data in graph_metrics.items():
                if isinstance(data, dict) and "values" in data:
                    m_grp = gm_grp.create_group(name)
                    if np.isscalar(data.get("mean")):
                        m_grp.attrs["mean"] = data["mean"]
                        m_grp.attrs["std"] = data["std"]
                        if "ci" in data:
                            m_grp.attrs["ci_low"] = data["ci"][0]
                            m_grp.attrs["ci_high"] = data["ci"][1]
                    m_grp.create_dataset("values", data=data["values"], **kw)

        # Hub results
        if hub_results is not None:
            hub_grp = f.create_group("hub_analysis")
            for k in [
                "betweenness_mean", "participation_mean",
                "within_module_z_mean", "hub_probability",
                "hub_class_mode", "hub_class_stability",
            ]:
                if k in hub_results:
                    hub_grp.create_dataset(k, data=hub_results[k])


def load_full_analysis(filepath: str) -> Dict:
    """
    Load full analysis from HDF5.

    Returns
    -------
    dict with keys: 'bootstrap_result', 'edge_classification',
        'community', 'graph_metrics', 'hub_analysis', 'metadata'.
    """
    import h5py

    output = {}
    output["bootstrap_result"] = load_bootstrap_result(filepath)

    with h5py.File(filepath, "r") as f:
        # Metadata
        output["metadata"] = dict(f["metadata"].attrs)

        # Edge classification
        if "edge_classification" in f:
            ec = f["edge_classification"]
            output["edge_classification"] = {
                "labels": ec["labels"][:],
                **{k: int(v) for k, v in ec.attrs.items()},
            }

        # Community
        if "community" in f:
            com = f["community"]
            output["community"] = {
                "coassignment": com["coassignment"][:],
                "consensus_partition": com["consensus_partition"][:],
                "node_stability": com["node_stability"][:],
            }
            if "all_partitions" in com:
                output["community"]["all_partitions"] = (
                    com["all_partitions"][:]
                )

        # Graph metrics
        if "graph_metrics" in f:
            gm = {}
            for name in f["graph_metrics"]:
                m = f["graph_metrics"][name]
                entry = {"values": m["values"][:]}
                for attr_name in ["mean", "std", "ci_low", "ci_high"]:
                    if attr_name in m.attrs:
                        entry[attr_name] = float(m.attrs[attr_name])
                gm[name] = entry
            output["graph_metrics"] = gm

        # Hub analysis
        if "hub_analysis" in f:
            hub = {}
            for name in f["hub_analysis"]:
                hub[name] = f["hub_analysis"][name][:]
            output["hub_analysis"] = hub

    return output


# =============================================================================
# BIDS-COMPATIBLE OUTPUT
# =============================================================================

def export_bids(
    result: BootstrapResult,
    output_dir: str,
    subject_id: str,
    session_id: Optional[str] = None,
    atlas_name: str = "Schaefer100",
    space: str = "MNI152NLin2009cAsym",
    edge_classification: Optional[Dict] = None,
    community_results: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Export results as BIDS connectivity derivatives.

    Follows BEP017 (Relationship/Connectivity Matrices) and BEP038
    (Atlas Specification) conventions.

    Output structure::

        output_dir/
        └── sub-{subject_id}/
            └── [ses-{session_id}/]
                └── dwi/
                    ├── sub-{id}_space-{space}_atlas-{atlas}_desc-mean_relmat.tsv
                    ├── sub-{id}_..._desc-std_relmat.tsv
                    ├── sub-{id}_..._desc-pnonzero_relmat.tsv
                    ├── sub-{id}_..._desc-cv_relmat.tsv
                    ├── sub-{id}_..._desc-edgeclass_relmat.tsv
                    ├── sub-{id}_..._desc-coassignment_relmat.tsv
                    └── sub-{id}_..._desc-bootstrap_relmat.json

    Parameters
    ----------
    result : BootstrapResult
    output_dir : str
    subject_id : str (without 'sub-' prefix)
    session_id : str, optional
    atlas_name : str
    space : str
    edge_classification : dict, optional
    community_results : dict, optional

    Returns
    -------
    dict mapping description to output filepath.
    """
    # Build BIDS path
    sub_str = f"sub-{subject_id}"
    parts = [sub_str]
    if session_id:
        parts.append(f"ses-{session_id}")
    parts.append("dwi")

    out_path = Path(output_dir) / "/".join(parts)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build filename prefix
    prefix_parts = [sub_str]
    if session_id:
        prefix_parts.append(f"ses-{session_id}")
    prefix_parts.extend([
        f"space-{space}",
        f"atlas-{atlas_name}",
    ])
    prefix = "_".join(prefix_parts)

    files = {}

    # SC matrices
    matrices = {
        "mean": result.sc_mean,
        "std": result.sc_std,
        "cv": result.sc_cv,
        "pnonzero": result.prob_nonzero,
        "cilow": result.sc_ci_low,
        "cihigh": result.sc_ci_high,
    }

    for desc, mat in matrices.items():
        fname = f"{prefix}_desc-{desc}_relmat.tsv"
        fpath = out_path / fname
        np.savetxt(fpath, mat, delimiter="\t", fmt="%.6f")
        files[desc] = str(fpath)

    # Edge classification
    if edge_classification is not None:
        fname = f"{prefix}_desc-edgeclass_relmat.tsv"
        fpath = out_path / fname
        np.savetxt(
            fpath, edge_classification["labels"],
            delimiter="\t", fmt="%d",
        )
        files["edgeclass"] = str(fpath)

    # Community co-assignment
    if community_results is not None:
        fname = f"{prefix}_desc-coassignment_relmat.tsv"
        fpath = out_path / fname
        np.savetxt(
            fpath, community_results["coassignment"],
            delimiter="\t", fmt="%.4f",
        )
        files["coassignment"] = str(fpath)

        # Consensus partition
        fname = f"{prefix}_desc-consensus_partition.tsv"
        fpath = out_path / fname
        np.savetxt(
            fpath, community_results["consensus_partition"],
            delimiter="\t", fmt="%d",
        )
        files["consensus"] = str(fpath)

    # JSON sidecar
    sidecar = {
        "n_bootstrap": result.n_bootstrap,
        "n_streamlines": result.n_streamlines,
        "n_parcels": result.n_parcels,
        "atlas": atlas_name,
        "space": space,
        "sift2_weighted": True,
        "software": "bootstracts",
        "software_version": __version__,
        "classification_thresholds": {
            "robust": ">95% presence AND CV < 0.5",
            "present": ">50% presence",
            "fragile": "5-50% presence",
            "spurious": "<5% presence",
        },
    }

    if edge_classification is not None:
        sidecar["n_robust"] = edge_classification.get("n_robust", 0)
        sidecar["n_present"] = edge_classification.get("n_present", 0)
        sidecar["n_fragile"] = edge_classification.get("n_fragile", 0)
        sidecar["n_spurious"] = edge_classification.get("n_spurious", 0)

    json_fname = f"{prefix}_desc-bootstrap_relmat.json"
    json_fpath = out_path / json_fname
    with open(json_fpath, "w") as jf:
        json.dump(sidecar, jf, indent=2)
    files["sidecar"] = str(json_fpath)

    return files

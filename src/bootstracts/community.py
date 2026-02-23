# -*- coding: utf-8 -*-
"""
bootstracts.community
==================================================

Probabilistic community detection and graph metrics with bootstrap CIs.

Instead of reporting a single partition or a single set of graph
metrics from one deterministic SC matrix, this module runs community
detection and metric computation across bootstrap SC samples, yielding:

1. **Co-assignment matrix**: P(i, j in same community) across all
   bootstrap resamples — more informative than any single partition.
2. **Node stability**: how consistently a node is assigned to the same
   module across resamples.  Low stability = boundary node.
3. **Consensus partition**: derived from the co-assignment matrix.
4. **Graph metric distributions**: density, strength, modularity,
   global/local efficiency, transitivity — all with 95% CIs.

Functions
---------
probabilistic_community_detection
    Community detection across bootstrap SC samples.
graph_metrics_with_ci
    Graph-theoretic metrics with bootstrap confidence intervals.

References
----------
- Rubinov & Sporns (2010). NeuroImage 52:1059-1069.
- Lancichinetti & Fortunato (2012). Sci Rep 2:336.
"""

import numpy as np
from typing import Dict

from .core import BootstrapResult, classify_edges


# =============================================================================
# LOUVAIN COMMUNITY DETECTION (pure numpy)
# =============================================================================

def _louvain_communities(
    sc: np.ndarray,
    resolution: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Louvain community detection (pure numpy implementation).

    Simplified but functional modularity optimization.
    For production use, consider replacing with ``community_louvain``
    from the ``community`` package (python-louvain).

    Parameters
    ----------
    sc : np.ndarray (N, N)
        Weighted adjacency matrix.
    resolution : float
        Resolution parameter γ.  Higher = more communities.
    seed : int

    Returns
    -------
    partition : np.ndarray (N,)
        Community assignment for each node.
    """
    rng = np.random.default_rng(seed)
    N = sc.shape[0]

    partition = np.arange(N)

    m = sc.sum() / 2.0
    if m == 0:
        return partition

    k = sc.sum(axis=1)

    improved = True
    max_iter = 50
    iteration = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        order = rng.permutation(N)

        for i in order:
            current_comm = partition[i]

            neighbor_comms = np.unique(partition[sc[i] > 0])
            if len(neighbor_comms) == 0:
                continue

            best_gain = 0
            best_comm = current_comm

            for c in neighbor_comms:
                if c == current_comm:
                    continue

                in_c = partition == c
                k_i_in = sc[i, in_c].sum()
                sigma_tot = k[in_c].sum()

                gain = k_i_in - resolution * k[i] * sigma_tot / (2 * m)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            if best_comm != current_comm:
                partition[i] = best_comm
                improved = True

    # Renumber to contiguous labels
    unique_comms = np.unique(partition)
    mapping = {old: new for new, old in enumerate(unique_comms)}
    partition = np.array([mapping[c] for c in partition])

    return partition


# =============================================================================
# PROBABILISTIC COMMUNITY DETECTION
# =============================================================================

def probabilistic_community_detection(
    result: BootstrapResult,
    n_community_runs: int = 100,
    resolution: float = 1.0,
    use_robust_only: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Community detection across bootstrap SC samples.

    For each of ``n_community_runs`` bootstrap samples, run Louvain
    community detection.  Then compute:

    1. **Co-assignment matrix**: P(i, j in same community).
    2. **Module stability per node**: how often this node stays in the
       same module.  High = confident assignment; low = boundary.
    3. **Consensus partition**: derived from the co-assignment matrix.

    Parameters
    ----------
    result : BootstrapResult
    n_community_runs : int
        Number of bootstrap samples to run community detection on.
        ``min(n_community_runs, n_bootstrap)`` will be used.
    resolution : float
        Louvain resolution parameter.
    use_robust_only : bool
        If True, zero out fragile/spurious edges before detection.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'coassignment' : np.ndarray (N, N)
            Co-assignment probability matrix.
        'consensus_partition' : np.ndarray (N,)
            Consensus community assignment.
        'node_stability' : np.ndarray (N,)
            Per-node stability (max pairwise co-assignment).
        'n_communities_distribution' : np.ndarray
            Number of communities per run.
        'modularity_distribution' : np.ndarray
            Modularity Q per run.
        'all_partitions' : np.ndarray (n_runs, N)
    """
    rng = np.random.default_rng(seed)
    N = result.n_parcels
    B = result.n_bootstrap

    n_runs = min(n_community_runs, B)

    if verbose:
        print(f"  Probabilistic community detection: {n_runs} runs")

    # Optional edge filtering
    if use_robust_only:
        edge_class = classify_edges(result)
        filter_mask = edge_class["present_mask"].astype(float)
    else:
        filter_mask = np.ones((N, N))

    coassignment = np.zeros((N, N))
    all_partitions = np.zeros((n_runs, N), dtype=int)
    n_communities = np.zeros(n_runs, dtype=int)
    modularities = np.zeros(n_runs)

    for r in range(n_runs):
        # Generate a bootstrap SC
        if result.sc_samples is not None and r < len(result.sc_samples):
            sc_b = result.sc_samples[r].astype(float)
        else:
            noise = rng.standard_normal((N, N))
            noise = (noise + noise.T) / 2.0
            sc_b = result.sc_mean + result.sc_std * noise
            sc_b = np.maximum(sc_b, 0)
            np.fill_diagonal(sc_b, 0)

        sc_b = sc_b * filter_mask

        # Detect communities
        partition = _louvain_communities(
            sc_b, resolution=resolution, seed=seed + r
        )
        all_partitions[r] = partition
        n_communities[r] = len(np.unique(partition))

        # Compute modularity Q
        m = sc_b.sum() / 2.0
        if m > 0:
            k = sc_b.sum(axis=1)
            Q = 0
            for c in np.unique(partition):
                in_c = partition == c
                Q += sc_b[np.ix_(in_c, in_c)].sum() / (2 * m)
                Q -= (k[in_c].sum() / (2 * m)) ** 2
            modularities[r] = Q

        # Update co-assignment matrix
        for c in np.unique(partition):
            in_c = np.where(partition == c)[0]
            for i in in_c:
                for j in in_c:
                    if i != j:
                        coassignment[i, j] += 1

        if verbose and (r + 1) % max(1, n_runs // 5) == 0:
            print(f"    {r+1}/{n_runs}")

    coassignment /= n_runs

    # Node stability: max pairwise co-assignment
    node_stability = np.zeros(N)
    for i in range(N):
        ca_i = coassignment[i].copy()
        ca_i[i] = 0
        if ca_i.max() > 0:
            node_stability[i] = ca_i.max()

    # Consensus partition from co-assignment matrix
    consensus = _louvain_communities(
        coassignment, resolution=resolution, seed=seed + 999
    )

    if verbose:
        n_cons_comm = len(np.unique(consensus))
        print(f"\n  Results:")
        print(f"    Consensus communities: {n_cons_comm}")
        print(
            f"    N communities range: "
            f"[{n_communities.min()}, {n_communities.max()}]"
            f" (median={np.median(n_communities):.0f})"
        )
        print(
            f"    Modularity Q: "
            f"{modularities.mean():.3f} ± {modularities.std():.3f}"
        )
        print(
            f"    Node stability: "
            f"{node_stability.mean():.3f} ± {node_stability.std():.3f}"
        )

        most_stable = np.argsort(node_stability)[-3:][::-1]
        least_stable = np.argsort(node_stability)[:3]
        print(
            f"    Most stable nodes: {most_stable} "
            f"(stability={node_stability[most_stable]})"
        )
        print(
            f"    Least stable nodes: {least_stable} "
            f"(stability={node_stability[least_stable]})"
        )

    return {
        "coassignment": coassignment,
        "consensus_partition": consensus,
        "node_stability": node_stability,
        "n_communities_distribution": n_communities,
        "modularity_distribution": modularities,
        "all_partitions": all_partitions,
    }


# =============================================================================
# GRAPH METRICS WITH UNCERTAINTY
# =============================================================================

def graph_metrics_with_ci(
    result: BootstrapResult,
    n_samples: int = 200,
    ci_level: float = 0.95,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Compute graph-theoretic metrics with bootstrap confidence intervals.

    Instead of reporting single-value metrics from one SC matrix,
    compute each metric across bootstrap samples and report the
    distribution.

    Metrics computed:
        - **Global**: density, mean strength, modularity Q,
          global efficiency, transitivity.
        - **Nodal**: node strength, local efficiency.

    Parameters
    ----------
    result : BootstrapResult
    n_samples : int
        Number of bootstrap samples to use (max = n_bootstrap).
    ci_level : float
    seed : int
    verbose : bool

    Returns
    -------
    dict
        ``{metric_name: {'mean', 'std', 'ci', 'values'}}``.
        For nodal metrics: values are ``(n_samples, N)`` arrays,
        with additional ``'ci_low'`` and ``'ci_high'`` keys.
    """
    rng = np.random.default_rng(seed)
    N = result.n_parcels
    n_samples = min(n_samples, result.n_bootstrap)
    alpha = (1 - ci_level) / 2

    if verbose:
        print(f"  Graph metrics with CI: {n_samples} bootstrap samples")

    global_metrics = {
        "density": np.zeros(n_samples),
        "mean_strength": np.zeros(n_samples),
        "strength_std": np.zeros(n_samples),
        "modularity": np.zeros(n_samples),
        "global_efficiency": np.zeros(n_samples),
        "transitivity": np.zeros(n_samples),
    }

    nodal_metrics = {
        "node_strength": np.zeros((n_samples, N)),
        "local_efficiency": np.zeros((n_samples, N)),
    }

    for s in range(n_samples):
        # Get or generate bootstrap SC
        if result.sc_samples is not None and s < len(result.sc_samples):
            sc = result.sc_samples[s].astype(float)
        else:
            noise = rng.standard_normal((N, N))
            noise = (noise + noise.T) / 2.0
            sc = result.sc_mean + result.sc_std * noise
            sc = np.maximum(sc, 0)
            np.fill_diagonal(sc, 0)

        # --- Global metrics ---
        binary = (sc > 0).astype(float)
        n_possible = N * (N - 1) / 2
        global_metrics["density"][s] = (
            binary[np.triu_indices(N, k=1)].sum() / n_possible
        )

        strengths = sc.sum(axis=1)
        global_metrics["mean_strength"][s] = strengths.mean()
        global_metrics["strength_std"][s] = strengths.std()

        # Modularity (from Louvain)
        partition = _louvain_communities(sc, seed=seed + s)
        m = sc.sum() / 2.0
        if m > 0:
            k = sc.sum(axis=1)
            Q = 0
            for c in np.unique(partition):
                in_c = partition == c
                Q += sc[np.ix_(in_c, in_c)].sum() / (2 * m)
                Q -= (k[in_c].sum() / (2 * m)) ** 2
            global_metrics["modularity"][s] = Q

        # Global efficiency (length = 1/weight)
        with np.errstate(divide="ignore"):
            dist = 1.0 / np.where(sc > 0, sc, np.inf)
        np.fill_diagonal(dist, 0)

        # Floyd-Warshall (feasible for N ≤ 300)
        sp = dist.copy()
        for k_node in range(N):
            sp = np.minimum(
                sp, sp[:, k_node:k_node + 1] + sp[k_node:k_node + 1, :]
            )

        with np.errstate(divide="ignore"):
            inv_sp = 1.0 / np.where(sp > 0, sp, np.inf)
        np.fill_diagonal(inv_sp, 0)
        global_metrics["global_efficiency"][s] = (
            inv_sp.sum() / (N * (N - 1))
        )

        # Transitivity (weighted)
        W_third = np.cbrt(sc)
        numerator = np.trace(W_third @ W_third @ W_third)
        denominator = (
            binary.sum(axis=1) * (binary.sum(axis=1) - 1)
        ).sum()
        if denominator > 0:
            global_metrics["transitivity"][s] = numerator / denominator

        # --- Nodal metrics ---
        nodal_metrics["node_strength"][s] = strengths

        for i in range(N):
            neighbors = np.where(sc[i] > 0)[0]
            if len(neighbors) < 2:
                continue
            sub_sc = sc[np.ix_(neighbors, neighbors)]
            n_neigh = len(neighbors)
            sub_dist = 1.0 / np.where(sub_sc > 0, sub_sc, np.inf)
            np.fill_diagonal(sub_dist, 0)

            for k_n in range(n_neigh):
                sub_dist = np.minimum(
                    sub_dist,
                    sub_dist[:, k_n:k_n + 1] + sub_dist[k_n:k_n + 1, :],
                )

            with np.errstate(divide="ignore"):
                inv_sub = 1.0 / np.where(sub_dist > 0, sub_dist, np.inf)
            np.fill_diagonal(inv_sub, 0)
            nodal_metrics["local_efficiency"][s, i] = (
                inv_sub.sum() / max(1, n_neigh * (n_neigh - 1))
            )

        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"    {s+1}/{n_samples}")

    # Compile results
    output = {}

    for name, values in global_metrics.items():
        output[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci": (
                float(np.percentile(values, alpha * 100)),
                float(np.percentile(values, (1 - alpha) * 100)),
            ),
            "values": values,
        }

    for name, values in nodal_metrics.items():
        node_means = values.mean(axis=0)
        node_stds = values.std(axis=0)
        node_ci_low = np.percentile(values, alpha * 100, axis=0)
        node_ci_high = np.percentile(values, (1 - alpha) * 100, axis=0)

        output[name] = {
            "mean": node_means,
            "std": node_stds,
            "ci_low": node_ci_low,
            "ci_high": node_ci_high,
            "values": values,
        }

    if verbose:
        print(f"\n  Global metrics (mean [95% CI]):")
        for name in global_metrics:
            m = output[name]
            print(
                f"    {name:>20s}: {m['mean']:.4f} "
                f"[{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]"
            )

    return output

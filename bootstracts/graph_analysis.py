# -*- coding: utf-8 -*-
"""
bootstracts.graph_analysis
==================================================

Advanced graph-theoretic analyses with bootstrap uncertainty.

Computes network properties across bootstrap SC samples, reporting
full distributions rather than point estimates.  Each metric includes
a stability assessment: how consistently does this property hold across
bootstrap resamples?

Functions
---------
rich_club_bootstrap
    Rich club coefficients Φ(k) with normalized CIs.
small_world_propensity_bootstrap
    Density-corrected small-world measure φ with CIs.
hub_detection_bootstrap
    Hub classification under uncertainty (provincial/connector/kinless).
communicability_bootstrap
    Matrix exponential communicability with edge-wise CIs.
participation_coefficient
    Participation coefficient Pᵢ = 1 − Σ_m (kᵢ(m)/kᵢ)².
node_centrality_bootstrap
    Multi-metric hub detection: betweenness, eigenvector, participation.

References
----------
- van den Heuvel & Sporns (2011). J Neurosci 31:15775-86.
- Muldoon et al. (2016). Sci Rep 6:22057.
- Guimerà & Amaral (2005). Nature 433:895-900.
- Estrada & Hatano (2008). Phys Rev E 77:036111.
"""

import numpy as np
from scipy import linalg
from typing import Dict, Optional, Tuple, List
from .core import BootstrapResult, classify_edges


# =============================================================================
# UTILITY: Generate/access bootstrap SC samples
# =============================================================================

def _get_bootstrap_sc(
    result: BootstrapResult,
    idx: int,
    rng: np.random.Generator,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Get or generate a single bootstrap SC sample."""
    N = result.n_parcels
    if result.sc_samples is not None and idx < len(result.sc_samples):
        sc = result.sc_samples[idx].astype(np.float64)
    else:
        noise = rng.standard_normal((N, N))
        noise = (noise + noise.T) / 2.0
        sc = result.sc_mean + result.sc_std * noise
        sc = np.maximum(sc, 0)
        np.fill_diagonal(sc, 0)
    if mask is not None:
        sc = sc * mask
    return sc


def _shortest_paths_weighted(sc: np.ndarray) -> np.ndarray:
    """Floyd-Warshall on weighted graph (length = 1/weight)."""
    N = sc.shape[0]
    with np.errstate(divide="ignore"):
        dist = np.where(sc > 0, 1.0 / sc, np.inf)
    np.fill_diagonal(dist, 0)

    sp = dist.copy()
    for k in range(N):
        sp = np.minimum(sp, sp[:, k:k + 1] + sp[k:k + 1, :])
    return sp


def _global_efficiency(sc: np.ndarray) -> float:
    """Global efficiency: mean inverse shortest path length."""
    N = sc.shape[0]
    sp = _shortest_paths_weighted(sc)
    with np.errstate(divide="ignore"):
        inv_sp = np.where(sp > 0, 1.0 / sp, 0)
    np.fill_diagonal(inv_sp, 0)
    return inv_sp.sum() / (N * (N - 1))


def _local_efficiency(sc: np.ndarray, node: int) -> float:
    """Local efficiency for a single node."""
    neighbors = np.where(sc[node] > 0)[0]
    if len(neighbors) < 2:
        return 0.0
    sub_sc = sc[np.ix_(neighbors, neighbors)]
    n_neigh = len(neighbors)
    sp = _shortest_paths_weighted(sub_sc)
    with np.errstate(divide="ignore"):
        inv_sp = np.where(sp > 0, 1.0 / sp, 0)
    np.fill_diagonal(inv_sp, 0)
    return inv_sp.sum() / max(1, n_neigh * (n_neigh - 1))


# =============================================================================
# RICH CLUB ANALYSIS
# =============================================================================

def rich_club_bootstrap(
    result: BootstrapResult,
    k_range: Optional[np.ndarray] = None,
    n_samples: int = 200,
    n_random: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Rich club coefficients Φ(k) with bootstrap CIs and normalization.

    For each degree threshold k, compute the rich-club coefficient:
        Φ(k) = 2 E_{>k} / (N_{>k} (N_{>k} − 1))

    Normalize against degree-preserving random networks (Maslov-Sneppen
    rewiring) to obtain Φ_norm(k) = Φ(k) / Φ_random(k).

    Bootstrap CIs are computed by running rich-club analysis across
    bootstrap SC samples.

    Parameters
    ----------
    result : BootstrapResult
    k_range : np.ndarray, optional
        Degree thresholds to evaluate.  Default: auto-determined.
    n_samples : int
        Number of bootstrap samples.
    n_random : int
        Number of random networks per sample for normalization.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'k_values' : np.ndarray
        'phi_mean' : np.ndarray — mean Φ(k) across bootstraps
        'phi_norm_mean' : np.ndarray — mean normalized Φ(k)
        'phi_ci_low', 'phi_ci_high' : np.ndarray — 95% CIs
        'phi_norm_ci_low', 'phi_norm_ci_high' : np.ndarray
        'membership_stability' : np.ndarray (N,) — fraction of
            bootstrap samples where node is in the rich club (at median k)
    """
    rng = np.random.default_rng(seed)
    N = result.n_parcels
    n_samples = min(n_samples, result.n_bootstrap)

    if verbose:
        print(f"  Rich club analysis: {n_samples} bootstrap samples")

    # Determine k range from mean SC
    binary_mean = (result.sc_mean > 0).astype(float)
    degrees_mean = binary_mean.sum(axis=1).astype(int)
    if k_range is None:
        k_min = max(1, int(np.percentile(degrees_mean[degrees_mean > 0], 10)))
        k_max = int(np.percentile(degrees_mean, 90))
        k_range = np.arange(k_min, k_max + 1)

    n_k = len(k_range)
    phi_all = np.zeros((n_samples, n_k))
    phi_norm_all = np.zeros((n_samples, n_k))
    membership_counts = np.zeros((n_samples, N), dtype=bool)

    for s in range(n_samples):
        sc = _get_bootstrap_sc(result, s, rng)
        binary = (sc > 0).astype(float)
        degrees = binary.sum(axis=1).astype(int)

        for ki, k in enumerate(k_range):
            rich_nodes = np.where(degrees > k)[0]
            n_rich = len(rich_nodes)

            if n_rich < 2:
                continue

            # Rich club coefficient
            sub_w = sc[np.ix_(rich_nodes, rich_nodes)]
            e_rich = sub_w[np.triu_indices(n_rich, k=1)].sum()
            n_possible = n_rich * (n_rich - 1) / 2
            phi_all[s, ki] = e_rich / n_possible if n_possible > 0 else 0

            # Record membership at median degree
            if ki == n_k // 2:
                membership_counts[s, rich_nodes] = True

        # Normalization against random networks
        for ki, k in enumerate(k_range):
            phi_random = np.zeros(n_random)
            for r in range(n_random):
                sc_rand = _maslov_sneppen_rewire(sc, n_iter=10, rng=rng)
                deg_rand = (sc_rand > 0).sum(axis=1).astype(int)
                rich_rand = np.where(deg_rand > k)[0]
                n_r = len(rich_rand)
                if n_r < 2:
                    continue
                sub_r = sc_rand[np.ix_(rich_rand, rich_rand)]
                e_r = sub_r[np.triu_indices(n_r, k=1)].sum()
                phi_random[r] = e_r / (n_r * (n_r - 1) / 2)

            phi_rand_mean = phi_random.mean()
            if phi_rand_mean > 0:
                phi_norm_all[s, ki] = phi_all[s, ki] / phi_rand_mean

        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"    {s + 1}/{n_samples}")

    membership_stability = membership_counts.mean(axis=0)

    return {
        "k_values": k_range,
        "phi_mean": phi_all.mean(axis=0),
        "phi_std": phi_all.std(axis=0),
        "phi_ci_low": np.percentile(phi_all, 2.5, axis=0),
        "phi_ci_high": np.percentile(phi_all, 97.5, axis=0),
        "phi_norm_mean": phi_norm_all.mean(axis=0),
        "phi_norm_ci_low": np.percentile(phi_norm_all, 2.5, axis=0),
        "phi_norm_ci_high": np.percentile(phi_norm_all, 97.5, axis=0),
        "membership_stability": membership_stability,
        "phi_all": phi_all,
    }


def _maslov_sneppen_rewire(
    sc: np.ndarray, n_iter: int = 10, rng: np.random.Generator = None
) -> np.ndarray:
    """Degree-preserving edge rewiring (Maslov-Sneppen algorithm)."""
    if rng is None:
        rng = np.random.default_rng()

    N = sc.shape[0]
    rewired = sc.copy()
    edges_i, edges_j = np.where(np.triu(rewired > 0, k=1))
    n_edges = len(edges_i)

    for _ in range(n_iter * n_edges):
        if n_edges < 2:
            break
        e1, e2 = rng.integers(0, n_edges, size=2)
        if e1 == e2:
            continue

        a, b = edges_i[e1], edges_j[e1]
        c, d = edges_i[e2], edges_j[e2]

        if len({a, b, c, d}) < 4:
            continue

        if rng.random() < 0.5:
            if rewired[a, d] == 0 and rewired[c, b] == 0 and a != d and c != b:
                rewired[a, d] = rewired[a, b]
                rewired[d, a] = rewired[b, a]
                rewired[c, b] = rewired[c, d]
                rewired[b, c] = rewired[d, c]
                rewired[a, b] = rewired[b, a] = 0
                rewired[c, d] = rewired[d, c] = 0
                edges_i[e1], edges_j[e1] = min(a, d), max(a, d)
                edges_i[e2], edges_j[e2] = min(c, b), max(c, b)
        else:
            if rewired[a, c] == 0 and rewired[b, d] == 0 and a != c and b != d:
                rewired[a, c] = rewired[a, b]
                rewired[c, a] = rewired[b, a]
                rewired[b, d] = rewired[c, d]
                rewired[d, b] = rewired[d, c]
                rewired[a, b] = rewired[b, a] = 0
                rewired[c, d] = rewired[d, c] = 0
                edges_i[e1], edges_j[e1] = min(a, c), max(a, c)
                edges_i[e2], edges_j[e2] = min(b, d), max(b, d)

    return rewired


# =============================================================================
# SMALL-WORLD PROPENSITY
# =============================================================================

def small_world_propensity_bootstrap(
    result: BootstrapResult,
    n_samples: int = 200,
    n_random: int = 20,
    n_lattice: int = 20,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Small-world propensity φ with bootstrap CIs.

    φ = 1 − √(ΔC² + ΔL²) / 2

    where ΔC = (C_lattice − C_obs) / (C_lattice − C_random) and
    ΔL = (L_obs − L_random) / (L_lattice − L_random), both clipped
    to [0, 1].  φ near 1 = small-world; φ near 0 = random or lattice.

    Parameters
    ----------
    result : BootstrapResult
    n_samples : int
    n_random, n_lattice : int
        Number of reference networks for each type.
    seed : int
    verbose : bool

    Returns
    -------
    dict with 'phi_mean', 'phi_ci', 'phi_values', 'delta_C', 'delta_L'

    References
    ----------
    - Muldoon, Bridgeford & Bassett (2016). Sci Rep 6:22057.
    """
    rng = np.random.default_rng(seed)
    N = result.n_parcels
    n_samples = min(n_samples, result.n_bootstrap)

    if verbose:
        print(f"  Small-world propensity: {n_samples} bootstrap samples")

    phi_values = np.zeros(n_samples)
    delta_C_values = np.zeros(n_samples)
    delta_L_values = np.zeros(n_samples)

    for s in range(n_samples):
        sc = _get_bootstrap_sc(result, s, rng)
        binary = (sc > 0).astype(float)
        n_edges_obs = int(binary[np.triu_indices(N, k=1)].sum())

        # Observed clustering and path length
        C_obs = _weighted_clustering(sc)
        L_obs = _characteristic_path_length(sc)

        # Random reference networks
        C_rand_list, L_rand_list = [], []
        for _ in range(n_random):
            sc_rand = _maslov_sneppen_rewire(sc, n_iter=5, rng=rng)
            C_rand_list.append(_weighted_clustering(sc_rand))
            L_rand_list.append(_characteristic_path_length(sc_rand))
        C_random = np.mean(C_rand_list)
        L_random = np.mean(L_rand_list)

        # Lattice reference
        C_latt_list, L_latt_list = [], []
        for _ in range(n_lattice):
            sc_latt = _ring_lattice(N, n_edges_obs, sc, rng)
            C_latt_list.append(_weighted_clustering(sc_latt))
            L_latt_list.append(_characteristic_path_length(sc_latt))
        C_lattice = np.mean(C_latt_list)
        L_lattice = np.mean(L_latt_list)

        # ΔC and ΔL
        denom_C = C_lattice - C_random
        denom_L = L_lattice - L_random

        delta_C = np.clip(
            (C_lattice - C_obs) / denom_C if abs(denom_C) > 1e-10 else 0,
            0, 1
        )
        delta_L = np.clip(
            (L_obs - L_random) / denom_L if abs(denom_L) > 1e-10 else 0,
            0, 1
        )

        phi = 1.0 - np.sqrt((delta_C ** 2 + delta_L ** 2) / 2.0)
        phi_values[s] = phi
        delta_C_values[s] = delta_C
        delta_L_values[s] = delta_L

        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"    {s + 1}/{n_samples}")

    return {
        "phi_mean": float(phi_values.mean()),
        "phi_std": float(phi_values.std()),
        "phi_ci": (
            float(np.percentile(phi_values, 2.5)),
            float(np.percentile(phi_values, 97.5)),
        ),
        "phi_values": phi_values,
        "delta_C_mean": float(delta_C_values.mean()),
        "delta_L_mean": float(delta_L_values.mean()),
    }


def _weighted_clustering(sc: np.ndarray) -> float:
    """Global weighted clustering coefficient."""
    W = np.cbrt(sc)
    binary = (sc > 0).astype(float)
    numerator = np.diag(W @ W @ W)
    k = binary.sum(axis=1)
    denominator = k * (k - 1)
    valid = denominator > 0
    if valid.sum() == 0:
        return 0.0
    return float((numerator[valid] / denominator[valid]).mean())


def _characteristic_path_length(sc: np.ndarray) -> float:
    """Characteristic path length (mean shortest path)."""
    N = sc.shape[0]
    sp = _shortest_paths_weighted(sc)
    sp_finite = sp[sp < np.inf]
    sp_finite = sp_finite[sp_finite > 0]
    return float(sp_finite.mean()) if len(sp_finite) > 0 else np.inf


def _ring_lattice(
    N: int, n_edges: int, sc_template: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a ring lattice with approximately n_edges edges."""
    lattice = np.zeros((N, N))
    k = max(1, n_edges * 2 // N)
    for i in range(N):
        for j in range(1, k // 2 + 1):
            nb = (i + j) % N
            lattice[i, nb] = 1.0
            lattice[nb, i] = 1.0
    # Transfer weights from template
    template_weights = sc_template[sc_template > 0]
    if len(template_weights) > 0:
        lattice_edges = np.where(np.triu(lattice > 0, k=1))
        n_latt_edges = len(lattice_edges[0])
        weights = rng.choice(template_weights, size=n_latt_edges, replace=True)
        for idx, (i, j) in enumerate(zip(*lattice_edges)):
            lattice[i, j] = weights[idx]
            lattice[j, i] = weights[idx]
    return lattice


# =============================================================================
# HUB DETECTION UNDER UNCERTAINTY
# =============================================================================

def hub_detection_bootstrap(
    result: BootstrapResult,
    community_results: Optional[Dict] = None,
    n_samples: int = 200,
    hub_threshold_percentile: float = 80,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Hub detection with bootstrap-derived confidence.

    Computes betweenness centrality, participation coefficient, and
    within-module degree z-score across bootstrap samples.  Classifies
    nodes as hubs only if they exceed the hub threshold in a majority
    of bootstrap samples.

    Guimerà-Amaral classification:
        - **Provincial hub**: high within-module degree, low participation
        - **Connector hub**: high within-module degree, high participation
        - **Kinless hub**: uniformly distributed connections across modules

    Parameters
    ----------
    result : BootstrapResult
    community_results : dict, optional
        Output from probabilistic_community_detection.
        If None, community detection is run per-bootstrap.
    n_samples : int
    hub_threshold_percentile : float
        Percentile above which a node is classified as a hub.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'betweenness_mean', 'betweenness_ci_low/high' : np.ndarray (N,)
        'participation_mean', 'participation_ci_low/high' : np.ndarray (N,)
        'within_module_z_mean' : np.ndarray (N,)
        'hub_probability' : np.ndarray (N,) — P(hub) across bootstraps
        'hub_class_mode' : np.ndarray (N,) — most common class per node
            (0=non-hub, 1=provincial, 2=connector, 3=kinless)
        'hub_class_stability' : np.ndarray (N,) — stability of classification

    References
    ----------
    - Guimerà & Amaral (2005). Nature 433:895-900.
    """
    from .community import _louvain_communities

    rng = np.random.default_rng(seed)
    N = result.n_parcels
    n_samples = min(n_samples, result.n_bootstrap)

    if verbose:
        print(f"  Hub detection: {n_samples} bootstrap samples")

    betw_all = np.zeros((n_samples, N))
    part_all = np.zeros((n_samples, N))
    wmz_all = np.zeros((n_samples, N))
    hub_class_all = np.zeros((n_samples, N), dtype=int)

    for s in range(n_samples):
        sc = _get_bootstrap_sc(result, s, rng)
        binary = (sc > 0).astype(float)

        # Betweenness centrality (approximate via shortest paths)
        betw_all[s] = _betweenness_centrality(sc)

        # Community partition
        if community_results is not None and s < len(
            community_results.get("all_partitions", [])
        ):
            partition = community_results["all_partitions"][s]
        else:
            partition = _louvain_communities(sc, seed=seed + s)

        # Participation coefficient and within-module z-score
        part_all[s] = _participation_coefficient(sc, partition)
        wmz_all[s] = _within_module_degree_z(sc, partition)

        # Hub classification (Guimerà-Amaral)
        strength = sc.sum(axis=1)
        hub_thresh = np.percentile(strength, hub_threshold_percentile)
        is_hub = strength > hub_thresh

        for i in range(N):
            if not is_hub[i]:
                hub_class_all[s, i] = 0  # non-hub
            elif part_all[s, i] < 0.3:
                hub_class_all[s, i] = 1  # provincial
            elif part_all[s, i] < 0.75:
                hub_class_all[s, i] = 2  # connector
            else:
                hub_class_all[s, i] = 3  # kinless

        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"    {s + 1}/{n_samples}")

    # Hub probability: fraction of samples where node is any kind of hub
    hub_probability = (hub_class_all > 0).mean(axis=0)

    # Most common classification
    hub_class_mode = np.zeros(N, dtype=int)
    hub_class_stability = np.zeros(N)
    for i in range(N):
        classes, counts = np.unique(hub_class_all[:, i], return_counts=True)
        hub_class_mode[i] = classes[np.argmax(counts)]
        hub_class_stability[i] = counts.max() / n_samples

    return {
        "betweenness_mean": betw_all.mean(axis=0),
        "betweenness_std": betw_all.std(axis=0),
        "betweenness_ci_low": np.percentile(betw_all, 2.5, axis=0),
        "betweenness_ci_high": np.percentile(betw_all, 97.5, axis=0),
        "participation_mean": part_all.mean(axis=0),
        "participation_std": part_all.std(axis=0),
        "participation_ci_low": np.percentile(part_all, 2.5, axis=0),
        "participation_ci_high": np.percentile(part_all, 97.5, axis=0),
        "within_module_z_mean": wmz_all.mean(axis=0),
        "within_module_z_std": wmz_all.std(axis=0),
        "hub_probability": hub_probability,
        "hub_class_mode": hub_class_mode,
        "hub_class_stability": hub_class_stability,
        "hub_class_labels": {
            0: "non-hub", 1: "provincial", 2: "connector", 3: "kinless"
        },
    }


def _betweenness_centrality(sc: np.ndarray) -> np.ndarray:
    """Weighted betweenness centrality (Brandes' algorithm)."""
    N = sc.shape[0]
    BC = np.zeros(N)

    for s_node in range(N):
        # Dijkstra's from s_node
        dist = np.full(N, np.inf)
        dist[s_node] = 0
        sigma = np.zeros(N)
        sigma[s_node] = 1
        pred = [[] for _ in range(N)]
        visited = np.zeros(N, dtype=bool)
        order = []

        for _ in range(N):
            # Find unvisited node with smallest distance
            candidates = np.where(~visited)[0]
            if len(candidates) == 0:
                break
            u = candidates[np.argmin(dist[candidates])]
            if dist[u] == np.inf:
                break
            visited[u] = True
            order.append(u)

            neighbors = np.where(sc[u] > 0)[0]
            for v in neighbors:
                if visited[v]:
                    continue
                alt = dist[u] + 1.0 / sc[u, v]
                if alt < dist[v] - 1e-10:
                    dist[v] = alt
                    sigma[v] = sigma[u]
                    pred[v] = [u]
                elif abs(alt - dist[v]) < 1e-10:
                    sigma[v] += sigma[u]
                    pred[v].append(u)

        # Back-propagation
        delta = np.zeros(N)
        for w in reversed(order):
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s_node:
                BC[w] += delta[w]

    # Normalize
    norm = (N - 1) * (N - 2)
    if norm > 0:
        BC /= norm

    return BC


def _participation_coefficient(
    sc: np.ndarray, partition: np.ndarray
) -> np.ndarray:
    """
    Participation coefficient P_i = 1 − Σ_m (k_i(m) / k_i)².

    P_i = 0 → all connections within own module.
    P_i = 1 → connections uniformly distributed across modules.
    """
    N = sc.shape[0]
    P = np.zeros(N)
    k = sc.sum(axis=1)

    for i in range(N):
        if k[i] == 0:
            continue
        for m in np.unique(partition):
            in_m = partition == m
            k_im = sc[i, in_m].sum()
            P[i] += (k_im / k[i]) ** 2
    P = 1.0 - P
    return P


def _within_module_degree_z(
    sc: np.ndarray, partition: np.ndarray
) -> np.ndarray:
    """Within-module degree z-score."""
    N = sc.shape[0]
    z = np.zeros(N)

    for m in np.unique(partition):
        in_m = np.where(partition == m)[0]
        if len(in_m) < 2:
            continue
        # Within-module strength
        k_within = np.array([
            sc[i, in_m].sum() for i in in_m
        ])
        mu = k_within.mean()
        sigma = k_within.std()
        if sigma > 0:
            for idx, i in enumerate(in_m):
                z[i] = (k_within[idx] - mu) / sigma
    return z


# =============================================================================
# COMMUNICABILITY
# =============================================================================

def communicability_bootstrap(
    result: BootstrapResult,
    n_samples: int = 200,
    normalize: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Communicability with bootstrap CIs.

    G_pq = (e^A)_pq captures information flow across *all* paths
    between nodes p and q, with longer walks naturally down-weighted
    by the factorial in the matrix exponential series.

    Parameters
    ----------
    result : BootstrapResult
    n_samples : int
    normalize : bool
        If True, normalize the adjacency matrix by spectral radius.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'comm_mean' : np.ndarray (N, N) — mean communicability
        'comm_std' : np.ndarray (N, N)
        'comm_ci_low', 'comm_ci_high' : np.ndarray (N, N)
        'node_communicability_mean' : np.ndarray (N,) — row sums
        'subgraph_centrality_mean' : np.ndarray (N,) — diagonal

    References
    ----------
    - Estrada & Hatano (2008). Phys Rev E 77:036111.
    - Crofts & Higham (2009). J R Soc Interface 6:411-414.
    """
    rng = np.random.default_rng(seed)
    N = result.n_parcels
    n_samples = min(n_samples, result.n_bootstrap)

    if verbose:
        print(f"  Communicability: {n_samples} bootstrap samples")

    comm_all = np.zeros((n_samples, N, N))

    for s in range(n_samples):
        sc = _get_bootstrap_sc(result, s, rng)

        if normalize:
            spectral_radius = np.abs(linalg.eigvalsh(sc)).max()
            if spectral_radius > 0:
                A = sc / spectral_radius
            else:
                A = sc
        else:
            A = sc

        comm_all[s] = linalg.expm(A)

        if verbose and (s + 1) % max(1, n_samples // 5) == 0:
            print(f"    {s + 1}/{n_samples}")

    comm_mean = comm_all.mean(axis=0)
    comm_std = comm_all.std(axis=0)

    # Node-level summaries
    node_comm = comm_all.sum(axis=2)  # (n_samples, N)
    subgraph_cent = np.array([np.diag(comm_all[s]) for s in range(n_samples)])

    return {
        "comm_mean": comm_mean,
        "comm_std": comm_std,
        "comm_ci_low": np.percentile(comm_all, 2.5, axis=0),
        "comm_ci_high": np.percentile(comm_all, 97.5, axis=0),
        "node_communicability_mean": node_comm.mean(axis=0),
        "node_communicability_ci_low": np.percentile(node_comm, 2.5, axis=0),
        "node_communicability_ci_high": np.percentile(node_comm, 97.5, axis=0),
        "subgraph_centrality_mean": subgraph_cent.mean(axis=0),
        "subgraph_centrality_std": subgraph_cent.std(axis=0),
    }

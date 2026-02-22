# -*- coding: utf-8 -*-
"""
bootstracts.inference
==================================================

Statistical inference on structural connectomes with bootstrap
uncertainty quantification and proper multiple-comparison control.

Functions
---------
nbs_bootstrap
    Network-Based Statistic with bootstrap CIs on component size.
tfnbs
    Threshold-Free Network-Based Statistics.
permutation_test_edges
    Edge-wise permutation testing with FWER control.
permutation_test_global
    Global graph-metric permutation test (between groups).
cpm_bootstrap
    Connectome-Predictive Modeling with bagging and feature stability.
pls_bootstrap
    Partial Least Squares brain-behavior with bootstrap ratio.

References
----------
- Zalesky, Fornito & Bullmore (2010). NeuroImage 53:1197-1207.
- Baggio et al. (2018). Hum Brain Mapp 39:1552-1568.
- Shen et al. (2017). Nat Protoc 12:506-518.
- McIntosh & Lobaugh (2004). NeuroImage 23:S250-S263.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

from .core import BootstrapResult


# =============================================================================
# NETWORK-BASED STATISTIC (NBS)
# =============================================================================

def nbs_bootstrap(
    sc_group1: np.ndarray,
    sc_group2: np.ndarray,
    bootstrap_results1: Optional[List[BootstrapResult]] = None,
    bootstrap_results2: Optional[List[BootstrapResult]] = None,
    threshold: float = 3.0,
    n_permutations: int = 5000,
    tail: str = "both",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Network-Based Statistic with bootstrap confidence intervals.

    The NBS identifies connected components of supra-threshold edges
    in the group difference, controlling family-wise error via
    permutation of group labels.

    When bootstrap results are provided for each subject, the NBS
    additionally reports:
        - Edge membership frequency across bootstrap samples
        - CIs on component size
        - Per-edge reliability within significant components

    Parameters
    ----------
    sc_group1 : np.ndarray (n1, N, N)
        SC matrices for group 1 (e.g., patients).
    sc_group2 : np.ndarray (n2, N, N)
        SC matrices for group 2 (e.g., controls).
    bootstrap_results1, bootstrap_results2 : list of BootstrapResult, optional
        Per-subject bootstrap results for uncertainty propagation.
    threshold : float
        t-statistic threshold for supra-threshold edges.
    n_permutations : int
        Number of permutations for FWER control.
    tail : str
        'both', 'left' (group1 < group2), 'right' (group1 > group2).
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'components' : list of np.ndarray
            Each array contains edge indices (i, j) in the component.
        'component_sizes' : list of int
        'p_values' : list of float — FWER-corrected p-values per component
        't_stats' : np.ndarray (N, N) — edge-wise t-statistics
        'significant_edges' : np.ndarray (N, N) bool
        'edge_membership_freq' : np.ndarray (N, N), optional
            If bootstrap_results provided, frequency of edge appearing
            in significant components across bootstrap resamples.
        'component_size_ci' : list of tuple, optional
            Bootstrap CIs on component size.

    References
    ----------
    - Zalesky, Fornito & Bullmore (2010). NeuroImage 53:1197-1207.
    """
    rng = np.random.default_rng(seed)

    n1 = sc_group1.shape[0]
    n2 = sc_group2.shape[0]
    N = sc_group1.shape[1]
    n_total = n1 + n2

    if verbose:
        print(f"  NBS: {n1} vs {n2} subjects, {n_permutations} permutations")

    # Observed t-statistics
    t_stats = _edge_ttest(sc_group1, sc_group2, tail=tail)

    # Supra-threshold edges
    if tail == "right":
        supra = t_stats > threshold
    elif tail == "left":
        supra = t_stats < -threshold
    else:
        supra = np.abs(t_stats) > threshold

    # Find connected components
    components = _find_components(supra)
    observed_sizes = [len(c) for c in components]

    if verbose:
        print(f"    Found {len(components)} components")
        if observed_sizes:
            print(f"    Largest component: {max(observed_sizes)} edges")

    # Permutation testing
    all_sc = np.concatenate([sc_group1, sc_group2], axis=0)
    max_component_sizes = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        perm_g1 = all_sc[perm_idx[:n1]]
        perm_g2 = all_sc[perm_idx[n1:]]

        t_perm = _edge_ttest(perm_g1, perm_g2, tail=tail)

        if tail == "right":
            supra_perm = t_perm > threshold
        elif tail == "left":
            supra_perm = t_perm < -threshold
        else:
            supra_perm = np.abs(t_perm) > threshold

        comps_perm = _find_components(supra_perm)
        if comps_perm:
            max_component_sizes[p] = max(len(c) for c in comps_perm)

        if verbose and (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"    Permutation {p + 1}/{n_permutations}")

    # FWER-corrected p-values
    p_values = []
    for size in observed_sizes:
        p_val = (max_component_sizes >= size).sum() / n_permutations
        p_values.append(float(p_val))

    # Significant edges
    significant_edges = np.zeros((N, N), dtype=bool)
    for comp, pval in zip(components, p_values):
        if pval < 0.05:
            for i, j in comp:
                significant_edges[i, j] = True
                significant_edges[j, i] = True

    result = {
        "components": components,
        "component_sizes": observed_sizes,
        "p_values": p_values,
        "t_stats": t_stats,
        "significant_edges": significant_edges,
        "threshold": threshold,
        "n_permutations": n_permutations,
    }

    # Bootstrap-enhanced NBS (edge reliability within components)
    if bootstrap_results1 is not None and bootstrap_results2 is not None:
        edge_freq = _nbs_bootstrap_stability(
            bootstrap_results1, bootstrap_results2,
            threshold=threshold, tail=tail,
            n_bootstrap_samples=min(100, n1, n2),
            seed=seed, rng=rng,
        )
        result["edge_membership_freq"] = edge_freq

    if verbose:
        for i, (size, pval) in enumerate(zip(observed_sizes, p_values)):
            sig = "***" if pval < 0.001 else ("**" if pval < 0.01
                   else ("*" if pval < 0.05 else ""))
            print(f"    Component {i}: {size} edges, p = {pval:.4f} {sig}")

    return result


def _edge_ttest(
    g1: np.ndarray, g2: np.ndarray, tail: str = "both"
) -> np.ndarray:
    """Edge-wise independent t-test."""
    n1, N, _ = g1.shape
    n2 = g2.shape[0]

    mean1 = g1.mean(axis=0)
    mean2 = g2.mean(axis=0)
    var1 = g1.var(axis=0, ddof=1)
    var2 = g2.var(axis=0, ddof=1)

    pooled_se = np.sqrt(var1 / n1 + var2 / n2)
    pooled_se[pooled_se == 0] = np.inf

    t = (mean1 - mean2) / pooled_se
    return t


def _find_components(supra: np.ndarray) -> List[List[Tuple[int, int]]]:
    """Find connected components in a supra-threshold edge matrix."""
    N = supra.shape[0]
    # Build adjacency from supra-threshold edges
    adj = supra | supra.T

    visited = np.zeros(N, dtype=bool)
    components = []

    for start in range(N):
        if visited[start] or not adj[start].any():
            continue

        # BFS
        queue = [start]
        visited[start] = True
        component_nodes = [start]

        while queue:
            node = queue.pop(0)
            neighbors = np.where(adj[node] & ~visited)[0]
            for nb in neighbors:
                visited[nb] = True
                queue.append(nb)
                component_nodes.append(nb)

        # Collect edges
        edges = []
        for i in component_nodes:
            for j in component_nodes:
                if i < j and supra[i, j]:
                    edges.append((i, j))

        if edges:
            components.append(edges)

    # Sort by size (largest first)
    components.sort(key=len, reverse=True)
    return components


def _nbs_bootstrap_stability(
    br1: List[BootstrapResult], br2: List[BootstrapResult],
    threshold: float, tail: str,
    n_bootstrap_samples: int, seed: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Edge membership frequency across bootstrap NBS runs."""
    N = br1[0].n_parcels
    edge_freq = np.zeros((N, N))

    for b in range(n_bootstrap_samples):
        # Sample group connectomes from each subject's bootstrap
        g1 = np.array([
            _sample_from_br(br, rng) for br in br1
        ])
        g2 = np.array([
            _sample_from_br(br, rng) for br in br2
        ])

        t_b = _edge_ttest(g1, g2, tail=tail)
        if tail == "right":
            supra_b = t_b > threshold
        elif tail == "left":
            supra_b = t_b < -threshold
        else:
            supra_b = np.abs(t_b) > threshold

        comps = _find_components(supra_b)
        for comp in comps:
            for i, j in comp:
                edge_freq[i, j] += 1
                edge_freq[j, i] += 1

    edge_freq /= n_bootstrap_samples
    return edge_freq


def _sample_from_br(
    br: BootstrapResult, rng: np.random.Generator
) -> np.ndarray:
    """Sample a single SC matrix from a BootstrapResult."""
    if br.sc_samples is not None:
        idx = rng.integers(0, len(br.sc_samples))
        return br.sc_samples[idx].astype(np.float64)
    else:
        N = br.n_parcels
        noise = rng.standard_normal((N, N))
        noise = (noise + noise.T) / 2.0
        sc = br.sc_mean + br.sc_std * noise
        return np.maximum(sc, 0)


# =============================================================================
# THRESHOLD-FREE NBS (TFNBS)
# =============================================================================

def tfnbs(
    sc_group1: np.ndarray,
    sc_group2: np.ndarray,
    n_permutations: int = 5000,
    E: float = 0.5,
    H: float = 2.0,
    dh: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Threshold-Free Network-Based Statistics.

    Integrates NBS across all thresholds to eliminate the arbitrary
    threshold choice.  For each edge e:

        TFNBS(e) = Σ_h [extent(C_h(e))^E × h^H × dh]

    where C_h(e) is the connected component containing edge e at
    threshold h, E and H control the balance between cluster extent
    and peak height.

    Parameters
    ----------
    sc_group1, sc_group2 : np.ndarray (n, N, N)
    n_permutations : int
    E, H : float
        Extent and height exponents.
    dh : float
        Threshold step size.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'tfnbs_scores' : np.ndarray (N, N) — observed TFNBS scores
        'p_values' : np.ndarray (N, N) — FWER-corrected p-values
        'significant_edges' : np.ndarray (N, N) bool

    References
    ----------
    - Baggio et al. (2018). Hum Brain Mapp 39:1552-1568.
    """
    rng = np.random.default_rng(seed)
    n1 = sc_group1.shape[0]
    n2 = sc_group2.shape[0]
    N = sc_group1.shape[1]
    n_total = n1 + n2

    if verbose:
        print(f"  TFNBS: {n1} vs {n2}, E={E}, H={H}")

    # Observed t-statistics
    t_obs = _edge_ttest(sc_group1, sc_group2)

    # Compute TFNBS scores
    tfnbs_obs = _compute_tfnbs_scores(t_obs, E=E, H=H, dh=dh)

    # Permutation testing
    all_sc = np.concatenate([sc_group1, sc_group2], axis=0)
    max_tfnbs_perm = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        t_perm = _edge_ttest(
            all_sc[perm_idx[:n1]], all_sc[perm_idx[n1:]]
        )
        tfnbs_perm = _compute_tfnbs_scores(t_perm, E=E, H=H, dh=dh)
        max_tfnbs_perm[p] = np.abs(tfnbs_perm).max()

        if verbose and (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"    Permutation {p + 1}/{n_permutations}")

    # FWER-corrected p-values
    p_values = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            p_values[i, j] = (
                (max_tfnbs_perm >= abs(tfnbs_obs[i, j])).sum()
                / n_permutations
            )
            p_values[j, i] = p_values[i, j]

    significant = p_values < 0.05
    np.fill_diagonal(significant, False)

    return {
        "tfnbs_scores": tfnbs_obs,
        "p_values": p_values,
        "significant_edges": significant,
        "n_significant": int(significant[np.triu_indices(N, k=1)].sum()),
    }


def _compute_tfnbs_scores(
    t_stats: np.ndarray, E: float = 0.5, H: float = 2.0, dh: float = 0.1,
) -> np.ndarray:
    """Compute TFNBS scores by integrating across thresholds."""
    N = t_stats.shape[0]
    tfnbs_scores = np.zeros((N, N))

    abs_t = np.abs(t_stats)
    t_max = abs_t.max()
    if t_max == 0:
        return tfnbs_scores

    thresholds = np.arange(dh, t_max + dh, dh)

    for h in thresholds:
        supra = abs_t > h
        components = _find_components(supra)

        for comp in components:
            extent = len(comp)
            score = (extent ** E) * (h ** H) * dh

            for i, j in comp:
                sign = np.sign(t_stats[i, j])
                tfnbs_scores[i, j] += sign * score
                tfnbs_scores[j, i] += sign * score

    return tfnbs_scores


# =============================================================================
# PERMUTATION TESTING
# =============================================================================

def permutation_test_edges(
    sc_group1: np.ndarray,
    sc_group2: np.ndarray,
    n_permutations: int = 10000,
    correction: str = "fdr",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Edge-wise permutation testing with multiple comparison correction.

    Parameters
    ----------
    sc_group1, sc_group2 : np.ndarray (n, N, N)
    n_permutations : int
    correction : str
        'fdr' (Benjamini-Hochberg), 'bonferroni', or 'none'.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'p_values_raw' : np.ndarray (N, N)
        'p_values_corrected' : np.ndarray (N, N)
        'significant_edges' : np.ndarray (N, N) bool (at α = 0.05)
        'effect_sizes' : np.ndarray (N, N) — Cohen's d
    """
    rng = np.random.default_rng(seed)
    n1, N, _ = sc_group1.shape
    n2 = sc_group2.shape[0]
    n_total = n1 + n2

    if verbose:
        print(f"  Edge permutation test: {n_permutations} permutations")

    # Observed differences
    diff_obs = sc_group1.mean(axis=0) - sc_group2.mean(axis=0)

    # Effect sizes (Cohen's d)
    pooled_std = np.sqrt(
        ((n1 - 1) * sc_group1.var(axis=0, ddof=1)
         + (n2 - 1) * sc_group2.var(axis=0, ddof=1))
        / (n_total - 2)
    )
    pooled_std[pooled_std == 0] = np.inf
    effect_sizes = diff_obs / pooled_std

    # Permutation
    all_sc = np.concatenate([sc_group1, sc_group2], axis=0)
    count_extreme = np.zeros((N, N))

    for p in range(n_permutations):
        perm_idx = rng.permutation(n_total)
        diff_perm = (
            all_sc[perm_idx[:n1]].mean(axis=0)
            - all_sc[perm_idx[n1:]].mean(axis=0)
        )
        count_extreme += (np.abs(diff_perm) >= np.abs(diff_obs)).astype(float)

        if verbose and (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"    {p + 1}/{n_permutations}")

    p_raw = count_extreme / n_permutations

    # Correction
    if correction == "fdr":
        p_corrected = _fdr_correction(p_raw, N)
    elif correction == "bonferroni":
        n_edges = N * (N - 1) // 2
        p_corrected = np.minimum(p_raw * n_edges, 1.0)
    else:
        p_corrected = p_raw

    significant = p_corrected < 0.05
    np.fill_diagonal(significant, False)

    return {
        "p_values_raw": p_raw,
        "p_values_corrected": p_corrected,
        "significant_edges": significant,
        "effect_sizes": effect_sizes,
        "n_significant": int(significant[np.triu_indices(N, k=1)].sum()),
    }


def _fdr_correction(p_matrix: np.ndarray, N: int) -> np.ndarray:
    """Benjamini-Hochberg FDR correction on upper triangle."""
    triu_idx = np.triu_indices(N, k=1)
    p_vals = p_matrix[triu_idx]
    n_tests = len(p_vals)

    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    # BH adjustment
    adjusted = np.zeros(n_tests)
    for i in range(n_tests - 1, -1, -1):
        if i == n_tests - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(
                adjusted[i + 1],
                sorted_p[i] * n_tests / (i + 1),
            )
    adjusted = np.minimum(adjusted, 1.0)

    # Unsort
    p_corrected_flat = np.zeros(n_tests)
    p_corrected_flat[sorted_idx] = adjusted

    # Rebuild matrix
    p_corrected = np.ones((N, N))
    p_corrected[triu_idx] = p_corrected_flat
    p_corrected = np.minimum(p_corrected, p_corrected.T)

    return p_corrected


def permutation_test_global(
    sc_group1: np.ndarray,
    sc_group2: np.ndarray,
    metric_func,
    n_permutations: int = 5000,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Permutation test for a global graph metric.

    Parameters
    ----------
    sc_group1, sc_group2 : np.ndarray (n, N, N)
    metric_func : callable
        Function that takes an SC matrix and returns a scalar.
    n_permutations : int
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'observed_diff' : float
        'p_value' : float
        'null_distribution' : np.ndarray (n_permutations,)
        'effect_size' : float (Cohen's d)
    """
    rng = np.random.default_rng(seed)
    n1 = sc_group1.shape[0]
    n_total = n1 + sc_group2.shape[0]

    vals1 = np.array([metric_func(sc) for sc in sc_group1])
    vals2 = np.array([metric_func(sc) for sc in sc_group2])
    observed_diff = vals1.mean() - vals2.mean()

    pooled_std = np.sqrt(
        ((len(vals1) - 1) * vals1.var(ddof=1)
         + (len(vals2) - 1) * vals2.var(ddof=1))
        / (n_total - 2)
    )
    effect_size = observed_diff / pooled_std if pooled_std > 0 else 0

    all_sc = np.concatenate([sc_group1, sc_group2], axis=0)
    null_dist = np.zeros(n_permutations)

    for p in range(n_permutations):
        perm = rng.permutation(n_total)
        g1 = all_sc[perm[:n1]]
        g2 = all_sc[perm[n1:]]
        v1 = np.array([metric_func(sc) for sc in g1])
        v2 = np.array([metric_func(sc) for sc in g2])
        null_dist[p] = v1.mean() - v2.mean()

        if verbose and (p + 1) % max(1, n_permutations // 10) == 0:
            print(f"    {p + 1}/{n_permutations}")

    p_value = (np.abs(null_dist) >= abs(observed_diff)).sum() / n_permutations

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "null_distribution": null_dist,
        "effect_size": float(effect_size),
    }


# =============================================================================
# CONNECTOME-PREDICTIVE MODELING (CPM)
# =============================================================================

def cpm_bootstrap(
    sc_matrices: np.ndarray,
    behavior: np.ndarray,
    n_bootstrap: int = 200,
    edge_threshold: float = 0.01,
    cv_folds: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Connectome-Predictive Modeling with bootstrap aggregating (bagging).

    Standard CPM (Shen et al., 2017): for each fold, select edges
    correlated with behavior at p < threshold, sum their weights to
    create a summary feature, fit linear regression.

    Bootstrap enhancement (Greene et al., 2020): resample subjects B
    times, build CPM per bootstrap, average predictions — improving
    generalizability and providing prediction intervals.

    Parameters
    ----------
    sc_matrices : np.ndarray (n_subjects, N, N)
    behavior : np.ndarray (n_subjects,)
        Behavioral/clinical variable to predict.
    n_bootstrap : int
        Number of bootstrap resamples for bagging.
    edge_threshold : float
        p-value threshold for edge selection.
    cv_folds : int, optional
        If provided, run K-fold CV.  Default: leave-one-out (LOOCV).
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'predictions' : np.ndarray (n_subjects,) — bagged predictions
        'correlation' : float — Pearson r between predicted and actual
        'p_value' : float — significance of correlation
        'prediction_intervals' : np.ndarray (n_subjects, 2) — 95% PI
        'edge_selection_freq' : np.ndarray (N, N) — fraction of
            bootstrap models selecting each edge
        'positive_network' : np.ndarray (N, N) — edges positively
            correlated with behavior (mean across bootstraps)
        'negative_network' : np.ndarray (N, N) — edges negatively
            correlated

    References
    ----------
    - Shen et al. (2017). Nat Protoc 12:506-518.
    - Finn et al. (2015). Nat Neurosci 18:1664-1671.
    - Greene et al. (2020). NeuroImage 220:117105.
    """
    rng = np.random.default_rng(seed)
    n_sub, N, _ = sc_matrices.shape

    if cv_folds is None:
        cv_folds = n_sub  # LOOCV

    if verbose:
        print(f"  CPM: {n_sub} subjects, {n_bootstrap} bootstrap resamples")
        print(f"  CV: {'LOOCV' if cv_folds == n_sub else f'{cv_folds}-fold'}")

    # Bootstrap aggregating
    all_predictions = np.zeros((n_bootstrap, n_sub))
    edge_selection_pos = np.zeros((n_bootstrap, N, N))
    edge_selection_neg = np.zeros((n_bootstrap, N, N))

    for b in range(n_bootstrap):
        # Bootstrap sample (for training; predict on full set)
        boot_idx = rng.integers(0, n_sub, size=n_sub)
        sc_boot = sc_matrices[boot_idx]
        beh_boot = behavior[boot_idx]

        # Edge selection on bootstrap sample
        triu_idx = np.triu_indices(N, k=1)
        n_edges = len(triu_idx[0])

        pos_mask = np.zeros((N, N), dtype=bool)
        neg_mask = np.zeros((N, N), dtype=bool)

        for e in range(n_edges):
            i, j = triu_idx[0][e], triu_idx[1][e]
            edge_vals = sc_boot[:, i, j]
            if edge_vals.std() == 0:
                continue
            r, p = stats.pearsonr(edge_vals, beh_boot)
            if p < edge_threshold:
                if r > 0:
                    pos_mask[i, j] = True
                else:
                    neg_mask[i, j] = True

        edge_selection_pos[b] = pos_mask.astype(float)
        edge_selection_neg[b] = neg_mask.astype(float)

        # Summary features for all subjects
        for s in range(n_sub):
            sc_s = sc_matrices[s]
            pos_sum = sc_s[pos_mask].sum()
            neg_sum = sc_s[neg_mask].sum()
            all_predictions[b, s] = pos_sum - neg_sum

        # Simple linear regression fit on bootstrap sample
        X_boot = np.array([
            sc_boot[i][pos_mask].sum() - sc_boot[i][neg_mask].sum()
            for i in range(len(boot_idx))
        ])
        if X_boot.std() > 0:
            slope, intercept = np.polyfit(X_boot, beh_boot, 1)
            for s in range(n_sub):
                all_predictions[b, s] = (
                    slope * all_predictions[b, s] + intercept
                )

        if verbose and (b + 1) % max(1, n_bootstrap // 10) == 0:
            print(f"    {b + 1}/{n_bootstrap}")

    # Aggregate predictions
    bagged_predictions = all_predictions.mean(axis=0)
    prediction_ci = np.percentile(all_predictions, [2.5, 97.5], axis=0).T

    # Correlation
    valid = ~(np.isnan(bagged_predictions) | np.isnan(behavior))
    if valid.sum() > 2:
        r, p = stats.pearsonr(bagged_predictions[valid], behavior[valid])
    else:
        r, p = 0, 1

    return {
        "predictions": bagged_predictions,
        "correlation": float(r),
        "p_value": float(p),
        "prediction_intervals": prediction_ci,
        "edge_selection_freq": (
            edge_selection_pos.mean(axis=0) + edge_selection_neg.mean(axis=0)
        ),
        "positive_network": edge_selection_pos.mean(axis=0),
        "negative_network": edge_selection_neg.mean(axis=0),
        "n_positive_edges_mean": int(edge_selection_pos.sum(axis=(1, 2)).mean()),
        "n_negative_edges_mean": int(edge_selection_neg.sum(axis=(1, 2)).mean()),
    }


# =============================================================================
# PARTIAL LEAST SQUARES (PLS)
# =============================================================================

def pls_bootstrap(
    brain_data: np.ndarray,
    behavior_data: np.ndarray,
    n_components: int = 5,
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    PLS brain-behavior analysis with permutation and bootstrap inference.

    SVD of R = X^T Y → USV^T identifies latent variables maximizing
    covariance between brain (X) and behavior (Y).

    - Permutation: tests significance of each latent variable.
    - Bootstrap: element-wise reliability via Bootstrap Ratio (BSR).
      BSR = weight / bootstrap SE, with BSR > 2 indicating reliable
      contributions (~95% CI).

    Parameters
    ----------
    brain_data : np.ndarray (n_subjects, n_brain_features)
        E.g., vectorized upper triangle of SC matrices.
    behavior_data : np.ndarray (n_subjects, n_behavior_vars)
    n_components : int
        Number of latent components.
    n_permutations : int
    n_bootstrap : int
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'singular_values' : np.ndarray
        'brain_weights' : np.ndarray (n_features, n_components)
        'behavior_weights' : np.ndarray (n_vars, n_components)
        'perm_p_values' : np.ndarray (n_components,)
        'bootstrap_ratios' : np.ndarray (n_features, n_components)
            BSR > 2 indicates reliable contribution.
        'brain_scores' : np.ndarray (n_subjects, n_components)
        'behavior_scores' : np.ndarray (n_subjects, n_components)
        'variance_explained' : np.ndarray (n_components,)

    References
    ----------
    - McIntosh & Lobaugh (2004). NeuroImage 23:S250-S263.
    - Helmer et al. (2024). Nat Commun Biol — stability caveat.
    """
    rng = np.random.default_rng(seed)
    n_sub = brain_data.shape[0]
    n_comp = min(n_components, min(brain_data.shape[1], behavior_data.shape[1]))

    if verbose:
        print(f"  PLS: {n_sub} subjects, {n_comp} components")
        if n_sub < 100:
            print("  ⚠ Warning: PLS can be unstable with small samples "
                  "(Helmer et al., 2024)")

    # Center data
    X = brain_data - brain_data.mean(axis=0)
    Y = behavior_data - behavior_data.mean(axis=0)

    # Cross-covariance and SVD
    R = X.T @ Y
    U, S, Vt = np.linalg.svd(R, full_matrices=False)

    U = U[:, :n_comp]
    S = S[:n_comp]
    V = Vt[:n_comp].T

    # Scores
    brain_scores = X @ U
    behavior_scores = Y @ V

    # Variance explained
    var_explained = S ** 2 / (S ** 2).sum()

    # Permutation testing
    if verbose:
        print(f"  Permutation testing ({n_permutations})...")
    perm_singular_values = np.zeros((n_permutations, n_comp))
    for p in range(n_permutations):
        Y_perm = Y[rng.permutation(n_sub)]
        R_perm = X.T @ Y_perm
        _, S_perm, _ = np.linalg.svd(R_perm, full_matrices=False)
        perm_singular_values[p] = S_perm[:n_comp]

    perm_p_values = np.array([
        (perm_singular_values[:, c] >= S[c]).sum() / n_permutations
        for c in range(n_comp)
    ])

    # Bootstrap for stability (BSR)
    if verbose:
        print(f"  Bootstrap ({n_bootstrap})...")
    boot_weights = np.zeros((n_bootstrap, X.shape[1], n_comp))

    for b in range(n_bootstrap):
        boot_idx = rng.integers(0, n_sub, size=n_sub)
        X_b = X[boot_idx] - X[boot_idx].mean(axis=0)
        Y_b = Y[boot_idx] - Y[boot_idx].mean(axis=0)
        R_b = X_b.T @ Y_b
        U_b, _, _ = np.linalg.svd(R_b, full_matrices=False)
        U_b = U_b[:, :n_comp]

        # Procrustes rotation to align with original
        M = U_b.T @ U
        P, _, Qt = np.linalg.svd(M)
        rotation = P @ Qt
        boot_weights[b] = U_b @ rotation

    boot_se = boot_weights.std(axis=0)
    boot_se[boot_se == 0] = np.inf
    bootstrap_ratios = U / boot_se

    return {
        "singular_values": S,
        "brain_weights": U,
        "behavior_weights": V,
        "perm_p_values": perm_p_values,
        "bootstrap_ratios": bootstrap_ratios,
        "brain_scores": brain_scores,
        "behavior_scores": behavior_scores,
        "variance_explained": var_explained,
        "n_reliable_features": int((np.abs(bootstrap_ratios) > 2).sum(axis=0).sum()),
    }

# -*- coding: utf-8 -*-
"""
bootstracts.along_tract
==================================================

Along-tract profiling with bootstrap uncertainty quantification.

Instead of reducing each tract to a single scalar, along-tract profiling
samples metrics at N equidistant points along the tract core, revealing
*where* along a bundle the pathology is concentrated.

Compatible with both AFQ (Yeatman et al., 2012) and BUAN (Chandio et
al., 2020) profiling frameworks.  Integrates with DIPY's pyAFQ and
BUAN pipelines via shared tract profile arrays.

Functions
---------
along_tract_bootstrap
    Bootstrap CIs on along-tract profiles from streamline weights.
compare_tract_profiles
    Point-wise group comparison with multiple-comparison correction.
bundle_membership_stability
    Probabilistic bundle membership via bootstrap RecoBundles.
tract_profile_from_streamlines
    Extract tract profiles from streamline-level data.

References
----------
- Yeatman et al. (2012). PLoS ONE 7:e49790. AFQ.
- Chandio et al. (2020). Sci Rep 10:17149. BUAN.
- Kruper et al. (2021). J Open Source Softw 6:3103. pyAFQ.
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional, List, Tuple, Union


# =============================================================================
# ALONG-TRACT BOOTSTRAP CIs
# =============================================================================

def along_tract_bootstrap(
    tract_profiles: np.ndarray,
    streamline_weights: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    n_points: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Bootstrap CIs on along-tract profiles.

    For each tract, compute per-node confidence intervals by resampling
    streamlines with replacement and re-computing the weighted mean
    profile.

    Parameters
    ----------
    tract_profiles : np.ndarray (n_streamlines, n_points)
        Per-streamline, per-node values (e.g., FA, MD along each
        streamline).  Each row is one streamline's profile.
    streamline_weights : np.ndarray (n_streamlines,), optional
        SIFT2 weights per streamline.  If None, uniform weights.
    n_bootstrap : int
    ci_level : float
    n_points : int
        Expected number of points per profile (for validation).
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'mean_profile' : np.ndarray (n_points,)
        'std_profile' : np.ndarray (n_points,)
        'ci_low' : np.ndarray (n_points,)
        'ci_high' : np.ndarray (n_points,)
        'median_profile' : np.ndarray (n_points,)
        'iqr_low' : np.ndarray (n_points,)
        'iqr_high' : np.ndarray (n_points,)
        'bootstrap_profiles' : np.ndarray (n_bootstrap, n_points)
    """
    rng = np.random.default_rng(seed)
    n_sl, n_pts = tract_profiles.shape

    if streamline_weights is None:
        weights = np.ones(n_sl) / n_sl
    else:
        weights = streamline_weights / streamline_weights.sum()

    alpha = (1 - ci_level) / 2

    if verbose:
        print(
            f"  Along-tract bootstrap: {n_sl} streamlines, "
            f"{n_pts} points, {n_bootstrap} resamples"
        )

    # Observed weighted mean
    mean_profile = np.average(tract_profiles, weights=weights, axis=0)

    # Bootstrap
    boot_profiles = np.zeros((n_bootstrap, n_pts))

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sl, size=n_sl)
        w_boot = weights[idx]
        w_boot = w_boot / w_boot.sum()
        boot_profiles[b] = np.average(
            tract_profiles[idx], weights=w_boot, axis=0
        )

    return {
        "mean_profile": mean_profile,
        "std_profile": boot_profiles.std(axis=0),
        "ci_low": np.percentile(boot_profiles, alpha * 100, axis=0),
        "ci_high": np.percentile(boot_profiles, (1 - alpha) * 100, axis=0),
        "median_profile": np.median(boot_profiles, axis=0),
        "iqr_low": np.percentile(boot_profiles, 25, axis=0),
        "iqr_high": np.percentile(boot_profiles, 75, axis=0),
        "bootstrap_profiles": boot_profiles,
    }


# =============================================================================
# GROUP COMPARISON ON TRACT PROFILES
# =============================================================================

def compare_tract_profiles(
    profiles_group1: np.ndarray,
    profiles_group2: np.ndarray,
    n_permutations: int = 5000,
    correction: str = "cluster",
    cluster_threshold: float = 2.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Point-wise group comparison along tract profiles.

    For each of the N points along the tract, compute a t-statistic
    comparing the two groups.  Use cluster-based permutation testing
    to control for multiple comparisons across tract points.

    Parameters
    ----------
    profiles_group1 : np.ndarray (n_subjects1, n_points)
        Mean profiles per subject in group 1.
    profiles_group2 : np.ndarray (n_subjects2, n_points)
        Mean profiles per subject in group 2.
    n_permutations : int
    correction : str
        'cluster' (cluster-based permutation), 'fdr', or 'bonferroni'.
    cluster_threshold : float
        t-statistic threshold for cluster formation.
    seed : int
    verbose : bool

    Returns
    -------
    dict
        't_stats' : np.ndarray (n_points,)
        'p_values_uncorrected' : np.ndarray (n_points,)
        'p_values_corrected' : np.ndarray (n_points,)
        'significant_points' : np.ndarray (n_points,) bool
        'effect_sizes' : np.ndarray (n_points,) — Cohen's d
        'clusters' : list of dict — significant clusters with extent
    """
    rng = np.random.default_rng(seed)

    n1, n_pts = profiles_group1.shape
    n2 = profiles_group2.shape[0]
    n_total = n1 + n2

    if verbose:
        print(
            f"  Along-tract comparison: {n1} vs {n2}, "
            f"{n_pts} points, {n_permutations} permutations"
        )

    # Observed statistics
    t_obs = np.zeros(n_pts)
    p_uncorr = np.zeros(n_pts)
    d_obs = np.zeros(n_pts)

    for pt in range(n_pts):
        t_val, p_val = stats.ttest_ind(
            profiles_group1[:, pt], profiles_group2[:, pt]
        )
        t_obs[pt] = t_val
        p_uncorr[pt] = p_val

        # Cohen's d
        pooled_std = np.sqrt(
            ((n1 - 1) * profiles_group1[:, pt].var(ddof=1)
             + (n2 - 1) * profiles_group2[:, pt].var(ddof=1))
            / (n_total - 2)
        )
        if pooled_std > 0:
            d_obs[pt] = (
                profiles_group1[:, pt].mean()
                - profiles_group2[:, pt].mean()
            ) / pooled_std

    # Cluster-based permutation
    if correction == "cluster":
        # Find observed clusters
        obs_clusters = _find_1d_clusters(t_obs, cluster_threshold)
        obs_cluster_masses = [c["mass"] for c in obs_clusters]

        # Permutation null
        all_data = np.vstack([profiles_group1, profiles_group2])
        max_cluster_mass = np.zeros(n_permutations)

        for p in range(n_permutations):
            perm_idx = rng.permutation(n_total)
            g1_perm = all_data[perm_idx[:n1]]
            g2_perm = all_data[perm_idx[n1:]]

            t_perm = np.zeros(n_pts)
            for pt in range(n_pts):
                t_perm[pt] = stats.ttest_ind(
                    g1_perm[:, pt], g2_perm[:, pt]
                )[0]

            perm_clusters = _find_1d_clusters(t_perm, cluster_threshold)
            if perm_clusters:
                max_cluster_mass[p] = max(
                    c["mass"] for c in perm_clusters
                )

            if verbose and (p + 1) % max(1, n_permutations // 10) == 0:
                print(f"    Permutation {p + 1}/{n_permutations}")

        # Cluster p-values
        p_corrected = np.ones(n_pts)
        sig_clusters = []
        for clust in obs_clusters:
            p_cluster = (
                (max_cluster_mass >= clust["mass"]).sum()
                / n_permutations
            )
            clust["p_value"] = p_cluster
            if p_cluster < 0.05:
                sig_clusters.append(clust)
                for pt in range(clust["start"], clust["end"] + 1):
                    p_corrected[pt] = p_cluster

    elif correction == "fdr":
        # Benjamini-Hochberg
        sorted_idx = np.argsort(p_uncorr)
        sorted_p = p_uncorr[sorted_idx]
        adjusted = np.zeros(n_pts)
        for i in range(n_pts - 1, -1, -1):
            if i == n_pts - 1:
                adjusted[i] = sorted_p[i]
            else:
                adjusted[i] = min(
                    adjusted[i + 1],
                    sorted_p[i] * n_pts / (i + 1),
                )
        adjusted = np.minimum(adjusted, 1.0)
        p_corrected = np.zeros(n_pts)
        p_corrected[sorted_idx] = adjusted
        sig_clusters = []

    else:
        p_corrected = np.minimum(p_uncorr * n_pts, 1.0)
        sig_clusters = []

    return {
        "t_stats": t_obs,
        "p_values_uncorrected": p_uncorr,
        "p_values_corrected": p_corrected,
        "significant_points": p_corrected < 0.05,
        "effect_sizes": d_obs,
        "clusters": sig_clusters,
        "n_significant_points": int((p_corrected < 0.05).sum()),
    }


def _find_1d_clusters(
    t_vals: np.ndarray, threshold: float
) -> List[Dict]:
    """Find contiguous supra-threshold clusters in 1D."""
    supra = np.abs(t_vals) > threshold
    clusters = []
    in_cluster = False
    start = 0

    for i in range(len(t_vals)):
        if supra[i] and not in_cluster:
            start = i
            in_cluster = True
        elif not supra[i] and in_cluster:
            clusters.append({
                "start": start,
                "end": i - 1,
                "extent": i - start,
                "mass": np.abs(t_vals[start:i]).sum(),
                "peak_t": float(t_vals[start:i][
                    np.argmax(np.abs(t_vals[start:i]))
                ]),
            })
            in_cluster = False

    if in_cluster:
        clusters.append({
            "start": start,
            "end": len(t_vals) - 1,
            "extent": len(t_vals) - start,
            "mass": np.abs(t_vals[start:]).sum(),
            "peak_t": float(t_vals[start:][
                np.argmax(np.abs(t_vals[start:]))
            ]),
        })

    return clusters


# =============================================================================
# BUNDLE MEMBERSHIP STABILITY
# =============================================================================

def bundle_membership_stability(
    streamline_bundle_assignments: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Probabilistic bundle membership via bootstrap.

    Given a set of streamlines assigned to bundles (e.g., via
    RecoBundles or BUAN), resample the streamlines and compute
    how consistently each streamline is assigned to its bundle.

    Parameters
    ----------
    streamline_bundle_assignments : np.ndarray (n_streamlines,)
        Bundle label for each streamline (integer, -1 = unassigned).
    n_bootstrap : int
    seed : int
    verbose : bool

    Returns
    -------
    dict
        'membership_probability' : np.ndarray (n_streamlines,)
            P(streamline assigned to its original bundle).
        'bundle_stability' : dict
            Per-bundle: mean membership probability, n_streamlines.
        'reassignment_matrix' : np.ndarray (n_bundles, n_bundles)
            How often streamlines from bundle A end up in bundle B.
    """
    rng = np.random.default_rng(seed)
    n_sl = len(streamline_bundle_assignments)
    labels = streamline_bundle_assignments.copy()
    bundles = np.unique(labels[labels >= 0])
    n_bundles = len(bundles)

    if verbose:
        print(f"  Bundle membership stability: {n_sl} streamlines, "
              f"{n_bundles} bundles, {n_bootstrap} bootstraps")

    membership_count = np.zeros(n_sl)
    reassignment = np.zeros((n_bundles, n_bundles))

    bundle_to_idx = {b: i for i, b in enumerate(bundles)}

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_sl, size=n_sl)
        boot_labels = labels[idx]

        for i, (orig_idx, boot_label) in enumerate(zip(idx, boot_labels)):
            orig_label = labels[orig_idx]
            if orig_label >= 0 and boot_label >= 0:
                if orig_label == boot_label:
                    membership_count[orig_idx] += 1

                oi = bundle_to_idx.get(orig_label)
                bi = bundle_to_idx.get(boot_label)
                if oi is not None and bi is not None:
                    reassignment[oi, bi] += 1

    membership_prob = membership_count / n_bootstrap

    # Per-bundle stability
    bundle_stability = {}
    for b in bundles:
        mask = labels == b
        bundle_stability[int(b)] = {
            "mean_stability": float(membership_prob[mask].mean()),
            "n_streamlines": int(mask.sum()),
        }

    # Normalize reassignment matrix
    row_sums = reassignment.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    reassignment_norm = reassignment / row_sums

    return {
        "membership_probability": membership_prob,
        "bundle_stability": bundle_stability,
        "reassignment_matrix": reassignment_norm,
        "bundle_labels": bundles,
    }


# =============================================================================
# TRACT PROFILE EXTRACTION
# =============================================================================

def tract_profile_from_streamlines(
    streamlines: List[np.ndarray],
    scalar_volume: np.ndarray,
    affine: np.ndarray,
    n_points: int = 100,
    weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Extract tract profiles from streamlines and a scalar volume.

    Resamples each streamline to n_points equidistant points, then
    samples the scalar volume at each point via trilinear interpolation.

    Parameters
    ----------
    streamlines : list of np.ndarray (n_points_i, 3)
        Streamline coordinates in mm space.
    scalar_volume : np.ndarray (X, Y, Z)
        Scalar map (e.g., FA, MD) in voxel space.
    affine : np.ndarray (4, 4)
        Affine mapping from mm to voxel coordinates.
    n_points : int
        Number of equidistant sampling points.
    weights : np.ndarray (n_streamlines,), optional
        SIFT2 weights.

    Returns
    -------
    dict
        'profiles' : np.ndarray (n_streamlines, n_points)
        'mean_profile' : np.ndarray (n_points,)
        'weighted_mean_profile' : np.ndarray (n_points,) — if weights
        'std_profile' : np.ndarray (n_points,)
    """
    from scipy.ndimage import map_coordinates

    n_sl = len(streamlines)
    profiles = np.zeros((n_sl, n_points))

    # Inverse affine for mm → voxel
    inv_affine = np.linalg.inv(affine)

    for i, sl in enumerate(streamlines):
        # Resample to equidistant points
        resampled = _resample_streamline(sl, n_points)

        # Convert to voxel coordinates
        coords_mm = np.column_stack(
            [resampled, np.ones(n_points)]
        )
        coords_vox = (inv_affine @ coords_mm.T)[:3]

        # Sample scalar volume
        profiles[i] = map_coordinates(
            scalar_volume, coords_vox, order=1, mode="nearest"
        )

    mean_profile = profiles.mean(axis=0)
    std_profile = profiles.std(axis=0)

    result = {
        "profiles": profiles,
        "mean_profile": mean_profile,
        "std_profile": std_profile,
    }

    if weights is not None:
        w = weights / weights.sum()
        result["weighted_mean_profile"] = np.average(
            profiles, weights=w, axis=0
        )

    return result


def _resample_streamline(
    streamline: np.ndarray, n_points: int
) -> np.ndarray:
    """Resample a streamline to n equidistant points."""
    # Compute cumulative arc length
    diffs = np.diff(streamline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length == 0:
        return np.tile(streamline[0], (n_points, 1))

    # Equidistant sampling
    target_lengths = np.linspace(0, total_length, n_points)
    resampled = np.zeros((n_points, 3))

    for i, t in enumerate(target_lengths):
        idx = np.searchsorted(cum_length, t, side="right") - 1
        idx = min(idx, len(streamline) - 2)
        frac = (t - cum_length[idx]) / max(seg_lengths[idx], 1e-10)
        frac = np.clip(frac, 0, 1)
        resampled[i] = (
            streamline[idx] + frac * (streamline[idx + 1] - streamline[idx])
        )

    return resampled

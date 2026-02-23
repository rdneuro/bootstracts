# -*- coding: utf-8 -*-
"""
bootstracts.voxel_bootstrap
==================================================

Voxel-level bootstrap methods for diffusion MRI.

These methods capture deeper uncertainty than tractogram-level bootstrap
by perturbing the diffusion signal itself, then re-fitting models and
re-tracking.  They are computationally expensive (hours per iteration)
but provide high-fidelity uncertainty estimates.

Functions
---------
wild_bootstrap_dti
    Wild bootstrap for DTI: handles heteroscedasticity in log-signal.
residual_bootstrap_csd
    Residual bootstrap for CSD: resamples SH model residuals.
voxel_bootstrap_connectome
    Full pipeline: voxel bootstrap → tracking → connectome → CIs.

References
----------
- Whitcher et al. (2008). Hum Brain Mapp 29:346-362.
- Jones (2008). IEEE Trans Med Imaging 27:1268-1274.
- Jeurissen et al. (2011). Hum Brain Mapp 32:461-479.
- Chung et al. (2006). NeuroImage 29:501-516.
"""

import numpy as np
from typing import Optional, Dict, Callable, Tuple
from dataclasses import dataclass, field


@dataclass
class VoxelBootstrapConfig:
    """
    Configuration for voxel-level bootstrap.

    Parameters
    ----------
    n_iterations : int
        Number of bootstrap iterations.  ≥500 recommended for CIs,
        ≥1000 for publication.
    model : str
        'dti' for wild bootstrap, 'csd' for residual bootstrap.
    hc_estimator : str
        Heteroscedasticity-consistent estimator for wild bootstrap.
        'hc2' or 'hc3' (recommended by Zhu et al. 2008).
    rademacher : bool
        If True (default), use Rademacher distribution {−1, +1}.
        If False, use Webb's 6-point distribution.
    seed : int
    """
    n_iterations: int = 1000
    model: str = "dti"
    hc_estimator: str = "hc3"
    rademacher: bool = True
    seed: int = 42


# =============================================================================
# WILD BOOTSTRAP FOR DTI
# =============================================================================

def wild_bootstrap_dti(
    dwi_data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[VoxelBootstrapConfig] = None,
    verbose: bool = True,
):
    """
    Wild bootstrap for diffusion tensor imaging.

    The DTI signal after log-transformation:
        y = X d + ε

    where X is the N_gradients × 7 design matrix encoding gradient
    directions.  Wild bootstrap:
        1. Fit OLS: d̂ = (X^T X)^{-1} X^T y
        2. Compute residuals: ê = y − X d̂
        3. Draw Rademacher variables t* ∈ {−1, +1}
        4. Construct: y* = X d̂ + ê ⊙ t*
        5. Refit and compute derived quantities

    The critical advantage is that multiplying each residual *in place*
    by a random sign preserves the location-dependent variance structure.

    Parameters
    ----------
    dwi_data : np.ndarray (..., n_gradients)
        DWI volumes.  Can be (X, Y, Z, n_gradients) or flattened.
    bvals : np.ndarray (n_gradients,)
    bvecs : np.ndarray (n_gradients, 3)
    mask : np.ndarray, optional
        Brain mask.
    config : VoxelBootstrapConfig, optional

    Yields
    ------
    tensor_field : np.ndarray (..., 6)
        Bootstrap tensor coefficients (D_xx, D_xy, D_xz, D_yy, D_yz, D_zz).

    References
    ----------
    - Whitcher et al. (2008). Hum Brain Mapp 29:346-362.
    - Jones (2008). IEEE Trans Med Imaging 27:1268-1274.
    """
    if config is None:
        config = VoxelBootstrapConfig(model="dti")

    rng = np.random.default_rng(config.seed)

    # Build DTI design matrix
    X = _build_dti_design_matrix(bvals, bvecs)
    n_grad = X.shape[0]

    # Reshape data
    original_shape = dwi_data.shape[:-1]
    data_2d = dwi_data.reshape(-1, n_grad)
    n_voxels = data_2d.shape[0]

    if mask is not None:
        mask_flat = mask.ravel()
        valid = mask_flat > 0
    else:
        valid = np.ones(n_voxels, dtype=bool)

    n_valid = valid.sum()

    if verbose:
        print(f"  Wild bootstrap DTI: {config.n_iterations} iterations")
        print(f"    Voxels: {n_valid:,} (of {n_voxels:,})")

    # Log-transform (handle zeros)
    y = np.log(np.maximum(data_2d[valid], 1e-6))  # (n_valid, n_grad)

    # OLS fit
    XtX_inv = np.linalg.pinv(X.T @ X)
    d_hat = (XtX_inv @ X.T @ y.T).T  # (n_valid, 7)
    y_hat = (X @ d_hat.T).T  # (n_valid, n_grad)
    residuals = y - y_hat  # (n_valid, n_grad)

    # HC leverage correction
    H = X @ XtX_inv @ X.T  # hat matrix
    h_diag = np.diag(H)

    if config.hc_estimator == "hc2":
        correction = 1.0 / np.sqrt(1.0 - h_diag)
    elif config.hc_estimator == "hc3":
        correction = 1.0 / (1.0 - h_diag)
    else:
        correction = np.ones(n_grad)

    corrected_residuals = residuals * correction[np.newaxis, :]

    # Bootstrap iterations
    for b in range(config.n_iterations):
        if config.rademacher:
            t_star = rng.choice([-1.0, 1.0], size=(n_valid, n_grad))
        else:
            # Webb's 6-point distribution
            vals = np.array([
                -np.sqrt(3/2), -1, -np.sqrt(1/2),
                 np.sqrt(1/2),  1,  np.sqrt(3/2),
            ])
            t_star = rng.choice(vals, size=(n_valid, n_grad))

        y_star = y_hat + corrected_residuals * t_star
        d_star = (XtX_inv @ X.T @ y_star.T).T

        # Extract tensor coefficients (skip S0 column)
        tensor_coeffs = d_star[:, 1:]  # (n_valid, 6)

        # Reconstruct full volume
        full_tensor = np.zeros((n_voxels, 6))
        full_tensor[valid] = tensor_coeffs

        yield full_tensor.reshape(*original_shape, 6)

        if verbose and (b + 1) % max(1, config.n_iterations // 10) == 0:
            print(f"    {b + 1}/{config.n_iterations}")


def _build_dti_design_matrix(
    bvals: np.ndarray, bvecs: np.ndarray
) -> np.ndarray:
    """
    Build DTI design matrix X (N_gradients × 7).

    Columns: [1, -b·gx², -b·gy², -b·gz², -2b·gx·gy, -2b·gx·gz, -2b·gy·gz]
    """
    n_grad = len(bvals)
    X = np.zeros((n_grad, 7))
    X[:, 0] = 1  # log(S0)

    for i in range(n_grad):
        b = bvals[i]
        g = bvecs[i]
        X[i, 1] = -b * g[0] ** 2
        X[i, 2] = -b * g[1] ** 2
        X[i, 3] = -b * g[2] ** 2
        X[i, 4] = -2 * b * g[0] * g[1]
        X[i, 5] = -2 * b * g[0] * g[2]
        X[i, 6] = -2 * b * g[1] * g[2]

    return X


# =============================================================================
# RESIDUAL BOOTSTRAP FOR CSD
# =============================================================================

def residual_bootstrap_csd(
    dwi_data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    response_function: np.ndarray,
    mask: Optional[np.ndarray] = None,
    lmax: int = 8,
    config: Optional[VoxelBootstrapConfig] = None,
    verbose: bool = True,
):
    """
    Residual bootstrap for constrained spherical deconvolution.

    CSD operates on untransformed signal where IID assumptions are more
    reasonable (Jeurissen et al., 2011).  The residual bootstrap:
        1. Fit CSD model → FOD coefficients
        2. Compute residuals
        3. Resample residuals with replacement
        4. Add resampled residuals to fitted signal
        5. Re-run CSD

    This captures uncertainty in fiber orientation estimation,
    propagating to downstream tractography.

    Parameters
    ----------
    dwi_data : np.ndarray (..., n_gradients)
    bvals : np.ndarray (n_gradients,)
    bvecs : np.ndarray (n_gradients, 3)
    response_function : np.ndarray
        Single-fiber response function (SH coefficients).
    mask : np.ndarray, optional
    lmax : int
        Maximum SH order.
    config : VoxelBootstrapConfig, optional

    Yields
    ------
    fod_field : np.ndarray (..., n_sh_coeffs)
        Bootstrap FOD coefficients.

    References
    ----------
    - Jeurissen et al. (2011). Hum Brain Mapp 32:461-479.
    """
    if config is None:
        config = VoxelBootstrapConfig(model="csd")

    rng = np.random.default_rng(config.seed)

    n_sh = int((lmax + 1) * (lmax + 2) / 2)

    # Build SH basis matrix
    B = _build_sh_basis(bvecs, lmax)
    n_grad = B.shape[0]

    # Build convolution matrix from response function
    R = _build_response_matrix(response_function, lmax, n_sh)

    # Design matrix for CSD: M = B @ R
    M = B @ R

    # Reshape data
    original_shape = dwi_data.shape[:-1]
    data_2d = dwi_data.reshape(-1, n_grad)
    n_voxels = data_2d.shape[0]

    if mask is not None:
        valid = mask.ravel() > 0
    else:
        valid = np.ones(n_voxels, dtype=bool)

    n_valid = valid.sum()

    if verbose:
        print(f"  Residual bootstrap CSD: {config.n_iterations} iterations")
        print(f"    Voxels: {n_valid:,}, lmax: {lmax}, SH coeffs: {n_sh}")

    y = data_2d[valid]  # (n_valid, n_grad)

    # OLS fit (approximation — full CSD uses non-negative constraint)
    M_pinv = np.linalg.pinv(M)
    fod_hat = (M_pinv @ y.T).T  # (n_valid, n_sh)
    y_hat = (M @ fod_hat.T).T  # (n_valid, n_grad)
    residuals = y - y_hat

    # Bootstrap iterations
    for b in range(config.n_iterations):
        # Resample residuals with replacement (per voxel)
        for v in range(n_valid):
            resample_idx = rng.integers(0, n_grad, size=n_grad)
            y_star_v = y_hat[v] + residuals[v, resample_idx]

            # Re-fit CSD (non-negative truncated)
            fod_star_v = M_pinv @ y_star_v
            fod_star_v = np.maximum(fod_star_v, 0)  # simple non-neg constraint

            if v == 0:
                fod_star = np.zeros((n_valid, n_sh))
            fod_star[v] = fod_star_v

        # Reconstruct full volume
        full_fod = np.zeros((n_voxels, n_sh))
        full_fod[valid] = fod_star

        yield full_fod.reshape(*original_shape, n_sh)

        if verbose and (b + 1) % max(1, config.n_iterations // 10) == 0:
            print(f"    {b + 1}/{config.n_iterations}")


def _build_sh_basis(bvecs: np.ndarray, lmax: int) -> np.ndarray:
    """
    Build real spherical harmonics basis matrix.

    Simplified implementation using Legendre polynomials.
    For production, use DIPY's `sph_harm_lookup`.
    """
    try:
        from scipy.special import sph_harm as _sph_harm
        _use_new_api = False
    except ImportError:
        from scipy.special import sph_harm_y
        _use_new_api = True

    n_grad = bvecs.shape[0]
    n_sh = int((lmax + 1) * (lmax + 2) / 2)
    B = np.zeros((n_grad, n_sh))

    # Convert Cartesian to spherical
    r = np.linalg.norm(bvecs, axis=1)
    r[r == 0] = 1
    theta = np.arccos(np.clip(bvecs[:, 2] / r, -1, 1))
    phi = np.arctan2(bvecs[:, 1], bvecs[:, 0])

    idx = 0
    for l_val in range(0, lmax + 1, 2):  # even orders only
        for m_val in range(-l_val, l_val + 1):
            if _use_new_api:
                # sph_harm_y(n, m, theta, phi) — theta=polar, phi=azimuthal
                Y = sph_harm_y(l_val, abs(m_val), theta, phi)
            else:
                # sph_harm(m, n, theta, phi) — theta=azimuthal, phi=polar
                Y = _sph_harm(abs(m_val), l_val, phi, theta)
            if m_val < 0:
                B[:, idx] = np.sqrt(2) * np.imag(Y)
            elif m_val == 0:
                B[:, idx] = np.real(Y)
            else:
                B[:, idx] = np.sqrt(2) * np.real(Y)
            idx += 1
            if idx >= n_sh:
                break
        if idx >= n_sh:
            break

    return B[:, :idx] if idx < n_sh else B


def _build_response_matrix(
    response: np.ndarray, lmax: int, n_sh: int
) -> np.ndarray:
    """Build diagonal convolution matrix from response function."""
    R = np.eye(n_sh)
    idx = 0
    for l_val in range(0, lmax + 1, 2):
        n_m = 2 * l_val + 1
        l_idx = l_val // 2
        if l_idx < len(response):
            R[idx:idx + n_m, idx:idx + n_m] *= response[l_idx]
        idx += n_m
    return R


# =============================================================================
# FULL VOXEL-BOOTSTRAP CONNECTOME PIPELINE
# =============================================================================

def voxel_bootstrap_connectome(
    dwi_data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    parcellation: np.ndarray,
    tracking_func: Callable,
    mask: Optional[np.ndarray] = None,
    config: Optional[VoxelBootstrapConfig] = None,
    n_parcels: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    Full voxel-bootstrap pipeline: perturb signal → track → connectome.

    This is the high-fidelity pathway that captures measurement
    uncertainty.  Each iteration:
        1. Generate bootstrap DWI via wild/residual bootstrap
        2. Fit diffusion model (DTI or CSD)
        3. Run tractography (user-provided function)
        4. Build connectome from the tractogram
        5. Accumulate statistics

    Parameters
    ----------
    dwi_data : np.ndarray (X, Y, Z, n_gradients)
    bvals, bvecs : np.ndarray
    parcellation : np.ndarray (X, Y, Z)
        Integer parcellation image.
    tracking_func : callable
        Function: (model_field, mask, parcellation) → sc_matrix (N, N).
        Must accept bootstrap model output and return a connectome.
    mask : np.ndarray, optional
    config : VoxelBootstrapConfig, optional
    n_parcels : int, optional
    verbose : bool

    Returns
    -------
    dict
        'sc_mean', 'sc_std', 'sc_cv' : np.ndarray (N, N)
        'prob_nonzero' : np.ndarray (N, N)
        'sc_ci_low', 'sc_ci_high' : np.ndarray (N, N)
        'n_iterations' : int

    Notes
    -----
    This function is computationally expensive. For a 2mm isotropic
    volume with 100k streamlines, each iteration may take 5-30 minutes.
    For clinical feasibility, consider running on a representative
    subset of bootstrap iterations (50-100) or using the faster
    tractogram-level bootstrap as the default pathway.
    """
    if config is None:
        config = VoxelBootstrapConfig()

    if n_parcels is None:
        n_parcels = int(parcellation.max())

    from .backends import WelfordAccumulator

    welford = WelfordAccumulator((n_parcels, n_parcels), track_nonzero=True)
    sc_all = []

    if verbose:
        print(f"  Voxel-bootstrap connectome: {config.n_iterations} iterations")
        print(f"  Model: {config.model}")
        print(f"  ⚠ This is computationally expensive!")

    if config.model == "dti":
        generator = wild_bootstrap_dti(
            dwi_data, bvals, bvecs, mask=mask,
            config=config, verbose=False,
        )
    else:
        raise NotImplementedError(
            "CSD voxel-bootstrap connectome requires a response function. "
            "Use residual_bootstrap_csd directly."
        )

    for b, model_field in enumerate(generator):
        # User-provided tracking + connectome building
        sc_b = tracking_func(model_field, mask, parcellation)
        welford.update(sc_b)
        sc_all.append(sc_b)

        if verbose and (b + 1) % max(1, config.n_iterations // 10) == 0:
            print(f"    {b + 1}/{config.n_iterations}")

    sc_mean, _, sc_std, prob_nonzero = welford.finalize()
    sc_cv = np.zeros_like(sc_mean)
    nz = sc_mean > 0
    sc_cv[nz] = sc_std[nz] / sc_mean[nz]

    sc_stack = np.array(sc_all)
    sc_ci_low = np.percentile(sc_stack, 2.5, axis=0)
    sc_ci_high = np.percentile(sc_stack, 97.5, axis=0)

    return {
        "sc_mean": sc_mean,
        "sc_std": sc_std,
        "sc_cv": sc_cv,
        "prob_nonzero": prob_nonzero,
        "sc_ci_low": sc_ci_low,
        "sc_ci_high": sc_ci_high,
        "n_iterations": config.n_iterations,
    }

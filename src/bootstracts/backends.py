# -*- coding: utf-8 -*-
"""
bootstracts.backends
==================================================

Tiered compute backend: NumPy → CuPy → JAX.

Provides a unified array interface so the rest of the library can
transparently use GPU acceleration when available.  Feature detection
is lazy: backends are only imported when requested.

Usage
-----
>>> from bootstracts.backends import get_backend
>>> xp = get_backend('cupy')  # falls back to numpy if unavailable
>>> arr = xp.zeros((100, 100))

The module also provides high-level bootstrap primitives that exploit
GPU parallelism for the inner loop (index generation + scatter
accumulation).
"""

import numpy as np
import warnings
from typing import Optional, Literal

# ── Backend registry ──────────────────────────────────────────────────

BackendName = Literal["numpy", "cupy", "jax"]

_BACKENDS = {}
_AVAILABLE = {}


def _probe_cupy() -> bool:
    try:
        import cupy as cp  # noqa: F401
        return True
    except ImportError:
        return False


def _probe_jax() -> bool:
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> dict:
    """Return dict of backend_name → bool availability."""
    if not _AVAILABLE:
        _AVAILABLE["numpy"] = True
        _AVAILABLE["cupy"] = _probe_cupy()
        _AVAILABLE["jax"] = _probe_jax()
    return dict(_AVAILABLE)


def get_backend(name: BackendName = "numpy"):
    """
    Return an array module (numpy-compatible API).

    Parameters
    ----------
    name : {'numpy', 'cupy', 'jax'}
        Requested backend.  Falls back to numpy with a warning
        if the requested backend is unavailable.

    Returns
    -------
    module
        Array module with numpy-compatible API.
    """
    if name in _BACKENDS:
        return _BACKENDS[name]

    avail = available_backends()

    if name == "cupy" and avail["cupy"]:
        import cupy as cp
        _BACKENDS["cupy"] = cp
        return cp
    elif name == "jax" and avail["jax"]:
        import jax.numpy as jnp
        _BACKENDS["jax"] = jnp
        return jnp
    elif name != "numpy":
        warnings.warn(
            f"Backend '{name}' not available, falling back to numpy.",
            RuntimeWarning,
            stacklevel=2,
        )

    _BACKENDS["numpy"] = np
    return np


# ── GPU-accelerated bootstrap primitives ──────────────────────────────

def bootstrap_sc_batch_gpu(
    parcel_a: np.ndarray,
    parcel_b: np.ndarray,
    weights: np.ndarray,
    n_parcels: int,
    batch_size: int = 100,
    seed: int = 42,
    backend: BackendName = "numpy",
) -> np.ndarray:
    """
    Generate a batch of bootstrap SC matrices.

    On GPU (CuPy), indices are generated and scatter-accumulated
    entirely on-device, transferring only the final matrices back.

    Parameters
    ----------
    parcel_a, parcel_b : np.ndarray (n_streamlines,)
        Streamline endpoint parcel indices.
    weights : np.ndarray (n_streamlines,)
        SIFT2 weights.
    n_parcels : int
    batch_size : int
        Number of bootstrap matrices to generate.
    seed : int
    backend : str

    Returns
    -------
    sc_batch : np.ndarray (batch_size, n_parcels, n_parcels)
        Bootstrap SC matrices (always returned as numpy).
    """
    xp = get_backend(backend)
    n_sl = len(parcel_a)

    if backend == "cupy" and available_backends()["cupy"]:
        import cupy as cp

        pa_d = cp.asarray(parcel_a)
        pb_d = cp.asarray(parcel_b)
        w_d = cp.asarray(weights)

        rng = cp.random.default_rng(seed)
        sc_batch = cp.zeros(
            (batch_size, n_parcels, n_parcels), dtype=cp.float32
        )

        for b in range(batch_size):
            idx = rng.integers(0, n_sl, size=n_sl)
            sc_b = cp.zeros((n_parcels, n_parcels), dtype=cp.float32)
            cp.add.at(sc_b, (pa_d[idx], pb_d[idx]), w_d[idx])
            sc_b = sc_b + sc_b.T
            cp.fill_diagonal(sc_b, 0)
            sc_batch[b] = sc_b

        return cp.asnumpy(sc_batch)

    elif backend == "jax" and available_backends()["jax"]:
        import jax
        import jax.numpy as jnp

        pa_j = jnp.array(parcel_a)
        pb_j = jnp.array(parcel_b)
        w_j = jnp.array(weights)

        def _single_bootstrap(key):
            idx = jax.random.randint(key, shape=(n_sl,), minval=0, maxval=n_sl)
            sc = jnp.zeros((n_parcels, n_parcels), dtype=jnp.float32)
            sc = sc.at[pa_j[idx], pb_j[idx]].add(w_j[idx])
            sc = sc + sc.T
            sc = sc.at[jnp.diag_indices(n_parcels)].set(0)
            return sc

        keys = jax.random.split(jax.random.PRNGKey(seed), batch_size)
        sc_batch = jax.vmap(_single_bootstrap)(keys)
        return np.asarray(sc_batch)

    else:
        # NumPy fallback
        rng = np.random.default_rng(seed)
        sc_batch = np.zeros(
            (batch_size, n_parcels, n_parcels), dtype=np.float32
        )
        for b in range(batch_size):
            idx = rng.integers(0, n_sl, size=n_sl)
            sc_b = np.zeros((n_parcels, n_parcels), dtype=np.float32)
            np.add.at(sc_b, (parcel_a[idx], parcel_b[idx]), weights[idx])
            sc_b = sc_b + sc_b.T
            np.fill_diagonal(sc_b, 0)
            sc_batch[b] = sc_b

        return sc_batch


# ── Welford's online algorithm ────────────────────────────────────────

class WelfordAccumulator:
    """
    Welford's streaming mean/variance for matrices.

    Avoids storing all B bootstrap samples by computing running
    statistics in a single pass.  Numerically stable for large B.

    Usage
    -----
    >>> acc = WelfordAccumulator(shape=(100, 100))
    >>> for b in range(1000):
    ...     sc_b = generate_bootstrap_sample()
    ...     acc.update(sc_b)
    >>> mean, var, std = acc.finalize()
    """

    def __init__(self, shape: tuple, track_nonzero: bool = True):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)
        self.track_nonzero = track_nonzero
        if track_nonzero:
            self.nonzero_count = np.zeros(shape, dtype=np.int64)

    def update(self, x: np.ndarray):
        """Incorporate a new sample."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        if self.track_nonzero:
            self.nonzero_count += (x > 0).astype(np.int64)

    def finalize(self):
        """
        Return (mean, variance, std).

        If track_nonzero, also returns prob_nonzero as fourth element.
        """
        if self.n < 2:
            var = np.zeros_like(self.mean)
        else:
            var = self.M2 / self.n  # population variance

        std = np.sqrt(np.maximum(var, 0))

        if self.track_nonzero:
            pnz = self.nonzero_count / max(1, self.n)
            return self.mean, var, std, pnz

        return self.mean, var, std

    @property
    def count(self) -> int:
        return self.n

# -*- coding: utf-8 -*-
"""
Test suite for bootstracts.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
import tempfile
import os


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def synthetic_sc():
    """Create a synthetic SC matrix for testing."""
    rng = np.random.default_rng(42)
    N = 20
    sc = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                sc[i, j] = sc[j, i] = rng.exponential(10)
    return sc


@pytest.fixture(scope="module")
def assignments(synthetic_sc):
    """Create synthetic streamline assignments."""
    from bootstracts.core import create_assignments_from_sc
    return create_assignments_from_sc(synthetic_sc, seed=42)


@pytest.fixture(scope="module")
def bootstrap_result(assignments):
    """Run bootstrap and return result."""
    from bootstracts.core import bootstrap_tractogram
    return bootstrap_tractogram(
        assignments, n_bootstrap=50, store_samples=True,
        seed=42, verbose=False,
    )


@pytest.fixture(scope="module")
def edge_classification(bootstrap_result):
    """Classify edges."""
    from bootstracts.core import classify_edges
    return classify_edges(bootstrap_result)


@pytest.fixture(scope="module")
def community_results(bootstrap_result):
    """Run community detection."""
    from bootstracts.community import probabilistic_community_detection
    return probabilistic_community_detection(
        bootstrap_result, n_community_runs=10,
        seed=42, verbose=False,
    )


@pytest.fixture(scope="module")
def group_data():
    """Create two groups of SC matrices for inference tests."""
    rng = np.random.default_rng(42)
    N = 20
    base = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < 0.3:
                base[i, j] = base[j, i] = rng.exponential(10)

    g1 = np.array([
        np.maximum(
            (base + (rng.normal(0, 0.3, (N, N))
                      + rng.normal(0, 0.3, (N, N)).T) / 2),
            0,
        )
        for _ in range(12)
    ])
    g2 = np.array([
        np.maximum(
            (base * 0.7 + (rng.normal(0, 0.3, (N, N))
                            + rng.normal(0, 0.3, (N, N)).T) / 2),
            0,
        )
        for _ in range(12)
    ])
    for i in range(12):
        np.fill_diagonal(g1[i], 0)
        np.fill_diagonal(g2[i], 0)

    return g1, g2


# =============================================================================
# CORE
# =============================================================================

class TestCore:
    def test_create_assignments(self, assignments):
        assert assignments.n_streamlines > 0
        assert assignments.n_parcels == 20
        assert len(assignments.parcel_a) == assignments.n_streamlines

    def test_bootstrap_tractogram(self, bootstrap_result):
        N = bootstrap_result.n_parcels
        assert bootstrap_result.sc_mean.shape == (N, N)
        assert bootstrap_result.sc_std.shape == (N, N)
        assert bootstrap_result.n_bootstrap == 50
        assert np.all(bootstrap_result.sc_mean >= 0)
        assert np.all(bootstrap_result.sc_std >= 0)

    def test_sc_observed(self, bootstrap_result):
        assert bootstrap_result.sc_observed is not None
        assert bootstrap_result.sc_observed.shape == (20, 20)

    def test_bootstrap_symmetry(self, bootstrap_result):
        assert np.allclose(
            bootstrap_result.sc_mean,
            bootstrap_result.sc_mean.T,
        )

    def test_classify_edges(self, edge_classification):
        ec = edge_classification
        total = ec["n_robust"] + ec["n_present"] + ec["n_fragile"] + ec["n_spurious"] + ec["n_absent"]
        N = 20
        expected = N * (N - 1) // 2
        assert total == expected

    def test_classify_edges_masks(self, edge_classification):
        ec = edge_classification
        labels = ec["labels"]
        assert labels.dtype == int
        assert np.all((labels >= 0) & (labels <= 4))

    def test_disparity_filter(self, bootstrap_result):
        from bootstracts.core import disparity_filter
        df = disparity_filter(bootstrap_result.sc_mean, alpha=0.1)
        assert "backbone" in df
        assert "significant_mask" in df
        assert df["backbone"].shape == bootstrap_result.sc_mean.shape
        assert 0 <= df["weight_retained_frac"] <= 1


# =============================================================================
# BACKENDS
# =============================================================================

class TestBackends:
    def test_numpy_backend(self):
        from bootstracts.backends import get_backend
        xp = get_backend("numpy")
        assert xp is not None

    def test_available_backends(self):
        from bootstracts.backends import available_backends
        backends = available_backends()
        assert "numpy" in backends

    def test_welford_accumulator(self):
        from bootstracts.backends import WelfordAccumulator
        rng = np.random.default_rng(42)
        wa = WelfordAccumulator((5, 5), track_nonzero=True)

        samples = [rng.random((5, 5)) for _ in range(100)]
        for s in samples:
            wa.update(s)

        mean, var, std, pnz = wa.finalize()
        expected_mean = np.mean(samples, axis=0)
        expected_std = np.std(samples, axis=0, ddof=0)

        assert wa.count == 100
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(std, expected_std, rtol=1e-5)


# =============================================================================
# COMMUNITY
# =============================================================================

class TestCommunity:
    def test_community_detection(self, community_results):
        cr = community_results
        assert "coassignment" in cr
        assert "consensus_partition" in cr
        assert "node_stability" in cr
        N = 20
        assert cr["coassignment"].shape == (N, N)
        assert len(cr["consensus_partition"]) == N
        assert len(cr["node_stability"]) == N

    def test_coassignment_range(self, community_results):
        ca = community_results["coassignment"]
        assert np.all(ca >= 0) and np.all(ca <= 1)

    def test_graph_metrics(self, bootstrap_result):
        from bootstracts.community import graph_metrics_with_ci
        gm = graph_metrics_with_ci(
            bootstrap_result, n_samples=10,
            seed=42, verbose=False,
        )
        for name in ["density", "mean_strength", "modularity",
                      "global_efficiency", "transitivity"]:
            assert name in gm
            assert "mean" in gm[name]
            assert "ci" in gm[name]
            assert gm[name]["ci"][0] <= gm[name]["mean"] <= gm[name]["ci"][1]


# =============================================================================
# GRAPH ANALYSIS
# =============================================================================

class TestGraphAnalysis:
    def test_rich_club(self, bootstrap_result):
        from bootstracts.graph_analysis import rich_club_bootstrap
        rc = rich_club_bootstrap(
            bootstrap_result, n_samples=5, n_random=3,
            seed=42, verbose=False,
        )
        assert "k_values" in rc
        assert "phi_mean" in rc
        assert "phi_norm_mean" in rc
        assert len(rc["k_values"]) > 0

    def test_hub_detection(self, bootstrap_result, community_results):
        from bootstracts.graph_analysis import hub_detection_bootstrap
        hubs = hub_detection_bootstrap(
            bootstrap_result, community_results=community_results,
            n_samples=5, seed=42, verbose=False,
        )
        assert "hub_probability" in hubs
        assert "participation_mean" in hubs
        assert "hub_class_mode" in hubs
        assert len(hubs["hub_probability"]) == 20

    def test_communicability(self, bootstrap_result):
        from bootstracts.graph_analysis import communicability_bootstrap
        cb = communicability_bootstrap(
            bootstrap_result, n_samples=5,
            seed=42, verbose=False,
        )
        assert "comm_mean" in cb
        assert "subgraph_centrality_mean" in cb
        assert cb["comm_mean"].shape == (20, 20)

    def test_small_world(self, bootstrap_result):
        from bootstracts.graph_analysis import small_world_propensity_bootstrap
        sw = small_world_propensity_bootstrap(
            bootstrap_result, n_samples=5,
            seed=42, verbose=False,
        )
        assert "phi_mean" in sw
        assert 0 <= sw["phi_mean"] <= 1


# =============================================================================
# INFERENCE
# =============================================================================

class TestInference:
    def test_nbs(self, group_data):
        from bootstracts.inference import nbs_bootstrap
        g1, g2 = group_data
        nbs = nbs_bootstrap(
            g1, g2, threshold=2.0, n_permutations=30,
            seed=42, verbose=False,
        )
        assert "significant_edges" in nbs
        assert "component_sizes" in nbs
        assert "p_values" in nbs

    def test_tfnbs(self, group_data):
        from bootstracts.inference import tfnbs
        g1, g2 = group_data
        tf = tfnbs(g1, g2, n_permutations=30, seed=42, verbose=False)
        assert "significant_edges" in tf
        assert "tfnbs_scores" in tf

    def test_permutation_test_edges(self, group_data):
        from bootstracts.inference import permutation_test_edges
        g1, g2 = group_data
        pte = permutation_test_edges(
            g1, g2, n_permutations=30,
            seed=42, verbose=False,
        )
        assert "p_values_raw" in pte
        assert "p_values_corrected" in pte
        assert "effect_sizes" in pte

    def test_permutation_test_global(self, group_data):
        from bootstracts.inference import permutation_test_global
        g1, g2 = group_data
        N = g1.shape[1]
        ptg = permutation_test_global(
            g1, g2,
            lambda sc: (sc > 0).sum() / (N * (N - 1)),
            n_permutations=30, seed=42, verbose=False,
        )
        assert "p_value" in ptg
        assert "observed_diff" in ptg
        assert 0 <= ptg["p_value"] <= 1

    def test_cpm(self, group_data):
        from bootstracts.inference import cpm_bootstrap
        g1, g2 = group_data
        all_sc = np.concatenate([g1, g2], axis=0)
        rng = np.random.default_rng(42)
        behavior = np.concatenate([
            rng.normal(50, 10, len(g1)),
            rng.normal(40, 10, len(g2)),
        ])
        cpm = cpm_bootstrap(
            all_sc, behavior, n_bootstrap=10,
            seed=42, verbose=False,
        )
        assert "predictions" in cpm
        assert "correlation" in cpm
        assert "edge_selection_freq" in cpm

    def test_pls(self, group_data):
        from bootstracts.inference import pls_bootstrap
        g1, g2 = group_data
        N = g1.shape[1]
        all_sc = np.concatenate([g1, g2], axis=0)
        X = np.array([s[np.triu_indices(N, k=1)] for s in all_sc])
        rng = np.random.default_rng(42)
        Y = np.concatenate([
            rng.normal(50, 10, len(g1)),
            rng.normal(40, 10, len(g2)),
        ]).reshape(-1, 1)
        pls = pls_bootstrap(
            X, Y, n_components=2,
            n_permutations=20, n_bootstrap=10,
            seed=42, verbose=False,
        )
        assert "singular_values" in pls
        assert "bootstrap_ratios" in pls
        assert "perm_p_values" in pls


# =============================================================================
# ALONG TRACT
# =============================================================================

class TestAlongTract:
    @pytest.fixture
    def tract_data(self):
        rng = np.random.default_rng(42)
        streamlines = [
            np.column_stack([
                np.linspace(10, 50, 30) + rng.normal(0, 0.5, 30),
                np.full(30, 30) + rng.normal(0, 0.3, 30),
                np.full(30, 30) + rng.normal(0, 0.3, 30),
            ])
            for _ in range(40)
        ]
        fa_vol = rng.uniform(0.2, 0.8, (60, 60, 60))
        affine = np.eye(4)
        return streamlines, fa_vol, affine

    def test_tract_profile(self, tract_data):
        from bootstracts.along_tract import tract_profile_from_streamlines
        sls, fa, aff = tract_data
        tp = tract_profile_from_streamlines(sls, fa, aff, n_points=20)
        assert tp["profiles"].shape == (40, 20)
        assert len(tp["mean_profile"]) == 20

    def test_along_tract_bootstrap(self, tract_data):
        from bootstracts.along_tract import (
            tract_profile_from_streamlines, along_tract_bootstrap,
        )
        sls, fa, aff = tract_data
        tp = tract_profile_from_streamlines(sls, fa, aff, n_points=20)
        atr = along_tract_bootstrap(
            tp["profiles"], n_bootstrap=20,
            n_points=20, seed=42, verbose=False,
        )
        assert "ci_low" in atr
        assert "ci_high" in atr
        assert "mean_profile" in atr

    def test_compare_profiles(self, tract_data):
        from bootstracts.along_tract import (
            tract_profile_from_streamlines, compare_tract_profiles,
        )
        sls, fa, aff = tract_data
        rng = np.random.default_rng(99)
        tp = tract_profile_from_streamlines(sls, fa, aff, n_points=20)
        g1 = tp["profiles"][:8] + rng.normal(0, 0.02, (8, 20))
        g2 = tp["profiles"][:8] * 0.9 + rng.normal(0, 0.02, (8, 20))
        comp = compare_tract_profiles(
            g1, g2, n_permutations=30,
            seed=42, verbose=False,
        )
        assert "significant_points" in comp
        assert "clusters" in comp
        assert "t_stats" in comp


# =============================================================================
# VOXEL BOOTSTRAP
# =============================================================================

class TestVoxelBootstrap:
    @pytest.fixture
    def dwi_data(self):
        rng = np.random.default_rng(42)
        n_vox, n_grad = 50, 20
        bvals = np.concatenate([[0] * 2, np.full(n_grad - 2, 1000)])
        bvecs = rng.standard_normal((n_grad, 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        dwi = np.abs(rng.normal(500, 50, (n_vox, n_grad)))
        return dwi, bvals, bvecs

    def test_wild_bootstrap_dti(self, dwi_data):
        from bootstracts.voxel_bootstrap import (
            VoxelBootstrapConfig, wild_bootstrap_dti,
        )
        dwi, bvals, bvecs = dwi_data
        config = VoxelBootstrapConfig(n_iterations=3)
        tensors = list(wild_bootstrap_dti(
            dwi, bvals, bvecs, config=config, verbose=False,
        ))
        assert len(tensors) == 3
        assert tensors[0].shape[0] == 50  # n_voxels

    def test_residual_bootstrap_csd(self, dwi_data):
        from bootstracts.voxel_bootstrap import (
            VoxelBootstrapConfig, residual_bootstrap_csd,
        )
        dwi, bvals, bvecs = dwi_data
        # CSD needs a response function (SH coefficients)
        response = np.array([1.0, 0.5, 0.1])
        config = VoxelBootstrapConfig(n_iterations=3, model="csd")
        fods = list(residual_bootstrap_csd(
            dwi, bvals, bvecs, response_function=response,
            config=config, verbose=False,
        ))
        assert len(fods) == 3


# =============================================================================
# STORAGE
# =============================================================================

class TestStorage:
    def test_save_load_bootstrap(self, bootstrap_result):
        from bootstracts.storage import (
            save_bootstrap_result, load_bootstrap_result,
        )
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            save_bootstrap_result(bootstrap_result, path)
            loaded = load_bootstrap_result(path)
            np.testing.assert_allclose(
                loaded.sc_mean, bootstrap_result.sc_mean,
            )
            assert loaded.n_bootstrap == bootstrap_result.n_bootstrap
        finally:
            os.unlink(path)

    def test_save_load_full(
        self, bootstrap_result, edge_classification,
        community_results,
    ):
        from bootstracts.storage import (
            save_full_analysis, load_full_analysis,
        )
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            save_full_analysis(
                path, bootstrap_result,
                edge_classification=edge_classification,
                community_results=community_results,
            )
            full = load_full_analysis(path)
            assert "bootstrap_result" in full
        finally:
            os.unlink(path)

    def test_bids_export(self, bootstrap_result):
        from bootstracts.storage import export_bids
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_bids(
                bootstrap_result, tmpdir,
                subject_id="sub01",
                atlas_name="Schaefer20",
            )
            # Check that files were created
            assert len(paths) > 0
            files = []
            for root, dirs, fnames in os.walk(tmpdir):
                files.extend(fnames)
            assert len(files) > 0
            assert any(f.endswith(".tsv") for f in files)
            assert any(f.endswith(".json") for f in files)


# =============================================================================
# IMPORT SMOKE TEST
# =============================================================================

class TestImports:
    def test_version(self):
        import bootstracts
        assert hasattr(bootstracts, "__version__")
        assert bootstracts.__version__ == "0.2.0"

    def test_all_exports(self):
        import bootstracts
        for name in bootstracts.__all__:
            assert hasattr(bootstracts, name), f"Missing export: {name}"

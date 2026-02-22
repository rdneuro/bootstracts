# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-22

### Added

- **GPU backends** (`backends.py`): Tiered NumPy → CuPy → JAX with automatic fallback
- **Welford streaming statistics**: O(1) memory for mean/variance during bootstrap
- **Disparity filter** (`core.py`): Serrano et al. (2009) multiscale backbone extraction
- **`sc_observed`** field in `BootstrapResult`: non-bootstrapped SC for comparison
- **Graph analysis** (`graph_analysis.py`):
  - Rich club coefficients Φ(k) with degree-preserving null models
  - Small-world propensity (Muldoon et al. 2016)
  - Hub detection with Guimerà-Amaral classification (provincial/connector/kinless)
  - Communicability via matrix exponential with bootstrap CIs
- **Statistical inference** (`inference.py`):
  - Network-Based Statistic (NBS) with FWER permutation control
  - Threshold-Free NBS (TFNBS, Baggio et al. 2018)
  - Edge-wise permutation testing with FDR/Bonferroni correction
  - Global metric permutation testing
  - Connectome-Predictive Modeling (CPM) with bootstrap aggregating
  - Partial Least Squares (PLS) brain-behavior with bootstrap ratios
- **Along-tract profiling** (`along_tract.py`):
  - Per-node bootstrap CIs on tract profiles
  - Group comparison with cluster-based permutation correction
  - Bundle membership stability via bootstrap RecoBundles
- **Voxel-level bootstrap** (`voxel_bootstrap.py`):
  - Wild bootstrap for DTI (Rademacher/Webb distributions, HC2/HC3)
  - Residual bootstrap for CSD (Jeurissen et al. 2011)
  - Full voxel-bootstrap connectome pipeline
- **Storage** (`storage.py`):
  - HDF5 save/load for all analysis types
  - BIDS-compatible export (BEP017/BEP038)
- **Extended visualizations** (`viz_extended.py`):
  - NBS results, rich club curves, hub cartography
  - Along-tract profiles, CPM predictions, communicability, PLS

## [0.1.0] - 2025-02-01

### Added

- Initial release as `sars.tractogram_bootstrap` module
- **Core bootstrap engine** (`core.py`):
  - `StreamlineAssignment`, `EdgeStats`, `BootstrapResult` data structures
  - MRtrix3 file loading (`tck2connectome -out_assignments`)
  - Synthetic assignment generation from SC matrices
  - Weighted tractogram bootstrap with running statistics
  - Edge reliability classification (robust/present/fragile/spurious)
- **Community detection** (`community.py`):
  - Probabilistic community detection across bootstrap samples
  - Co-assignment matrices and consensus partition
  - Node stability metrics
  - Graph metrics with bootstrap CIs (density, strength, modularity, efficiency, transitivity)
- **Visualization** (`viz.py`):
  - SC uncertainty maps (mean, std, CV)
  - Edge classification matrices
  - Community results (co-assignment, stability)
  - Graph metric distributions
  - Clinical correlation plots

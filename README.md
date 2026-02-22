# bootstracts

**Probabilistic structural connectomics via tractogram bootstrap.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Standard connectomics pipelines treat structural connectivity (SC) as deterministic:
each edge weight is a single number. But tractography is stochastic — seeding, tracking,
and filtering all introduce variability — and the uncertainty is real but universally ignored.

**bootstracts** quantifies that uncertainty via weighted bootstrap of the SIFT2 tractogram,
enabling confidence intervals on every SC edge, edge reliability classification, probabilistic
community detection, graph metric distributions, and network-level inference — all from a
single acquisition.

## Key Features

- **Tractogram bootstrap** — resample (streamline, SIFT2 weight) pairs to build SC distributions
- **Edge classification** — robust / present / fragile / spurious based on P(nonzero) and CV
- **Probabilistic community detection** — co-assignment matrices, consensus partition, node stability
- **Graph metrics with CIs** — density, strength, modularity, efficiency, transitivity
- **Network inference** — NBS, TFNBS, permutation testing (edge-wise and global)
- **Rich club, small-world, hub analysis** — Guimerà-Amaral cartography under uncertainty
- **Communicability** — matrix exponential with bootstrap CIs
- **CPM with bagging** — connectome-predictive modeling with prediction intervals
- **PLS brain-behavior** — bootstrap ratios for feature reliability
- **Along-tract profiling** — per-node CIs with cluster-based correction
- **Voxel-level bootstrap** — wild bootstrap (DTI) and residual bootstrap (CSD)
- **GPU acceleration** — NumPy → CuPy → JAX tiered backend
- **HDF5 + BIDS export** — BEP017/BEP038-compliant derivatives

## Installation

```bash
pip install bootstracts
```

With GPU support:

```bash
pip install bootstracts[gpu]    # CuPy backend
pip install bootstracts[jax]    # JAX backend
pip install bootstracts[all]    # everything + nilearn + dipy
```

For development:

```bash
git clone https://github.com/rdneuro/bootstracts.git
cd bootstracts
pip install -e ".[dev]"
```

## Quick Start

### From MRtrix3 outputs

```python
from bootstracts import (
    load_streamline_assignments,
    bootstrap_tractogram,
    classify_edges,
    probabilistic_community_detection,
    graph_metrics_with_ci,
    plot_sc_uncertainty,
    plot_edge_classification,
)

# Load pre-computed assignments from MRtrix3
# tckgen → tcksift2 → tck2connectome -out_assignments
assignments = load_streamline_assignments(
    connectome_csv="connectome.csv",
    weights_csv="sift2_weights.csv",
    assignments_txt="assignments.txt",
    n_parcels=100,
)

# Bootstrap: ~1000 resamples in seconds
result = bootstrap_tractogram(assignments, n_bootstrap=1000)

# Edge reliability classification
edge_class = classify_edges(result)
# → robust (>95% present, CV < 0.5), present, fragile, spurious

# Probabilistic community detection
communities = probabilistic_community_detection(result)
# → co-assignment matrix, consensus partition, node stability

# Graph metrics with 95% CIs
metrics = graph_metrics_with_ci(result)
# → density, strength, modularity, efficiency — all with distributions

# Publication-quality figures
plot_sc_uncertainty(result, save_path="fig1_uncertainty.png")
plot_edge_classification(edge_class, save_path="fig2_edges.png")
```

### Prototyping without raw tractograms

```python
from bootstracts import create_assignments_from_sc
import numpy as np

# Start from any SC matrix
sc = np.loadtxt("my_connectome.csv", delimiter=",")
assignments = create_assignments_from_sc(sc, expand_factor=20)
result = bootstrap_tractogram(assignments, n_bootstrap=1000)
```

### Group comparison (NBS)

```python
from bootstracts import nbs_bootstrap
import numpy as np

# patients: (n_patients, N, N), controls: (n_controls, N, N)
patients = np.stack([...])
controls = np.stack([...])

nbs = nbs_bootstrap(
    patients, controls,
    threshold=3.0,
    n_permutations=5000,
)
# → significant components with FWER-corrected p-values
```

### Hub analysis under uncertainty

```python
from bootstracts import hub_detection_bootstrap

hubs = hub_detection_bootstrap(
    result,
    community_results=communities,
    n_samples=200,
)
# → hub probability, Guimerà-Amaral classification stability,
#   betweenness/participation/within-module-z with CIs
```

### Brain-behavior prediction (CPM)

```python
from bootstracts import cpm_bootstrap

cpm = cpm_bootstrap(
    sc_matrices,       # (n_subjects, N, N)
    behavior_scores,   # (n_subjects,)
    n_bootstrap=200,
)
# → predictions with intervals, edge selection frequency
```

### Saving and BIDS export

```python
from bootstracts import save_full_analysis, export_bids

# Save everything to HDF5
save_full_analysis(
    "results.h5", result,
    edge_classification=edge_class,
    community_results=communities,
    graph_metrics=metrics,
    hub_results=hubs,
)

# BIDS-compatible derivatives (BEP017/BEP038)
export_bids(
    result, "derivatives/",
    subject_id="COVID001",
    atlas_name="Schaefer100",
)
```

## Architecture

```
bootstracts/
├── core.py             # Bootstrap engine, edge classification, disparity filter
├── backends.py         # GPU backends (NumPy/CuPy/JAX), Welford streaming
├── community.py        # Probabilistic community detection, graph metrics
├── graph_analysis.py   # Rich club, small-world, hubs, communicability
├── inference.py        # NBS, TFNBS, permutation tests, CPM, PLS
├── along_tract.py      # Tract profiling, cluster correction, bundle stability
├── voxel_bootstrap.py  # Wild bootstrap (DTI), residual bootstrap (CSD)
├── storage.py          # HDF5 I/O, BIDS export
├── viz.py              # Core visualizations (5 plot types)
└── viz_extended.py     # Advanced visualizations (7 plot types)
```

## MRtrix3 Integration

bootstracts reads directly from MRtrix3 pipeline outputs:

```bash
# 1. Generate tractogram
tckgen wmfod.mif tracks.tck -seed_image mask.nii.gz -select 10M

# 2. SIFT2 filtering (produces per-streamline weights)
tcksift2 tracks.tck wmfod.mif tracks_sift2.tck \
    -out_mu mu.txt -csv_output sift2_weights.csv

# 3. Build connectome with streamline assignments
tck2connectome tracks_sift2.tck parcellation.nii.gz connectome.csv \
    -out_assignments assignments.txt
```

## Dependencies

**Required:** NumPy ≥ 1.22, SciPy ≥ 1.9, Matplotlib ≥ 3.5, h5py ≥ 3.7

**Optional:** CuPy (GPU), JAX (GPU/TPU), nilearn, DIPY

## References

- Tournier et al. (2019). *NeuroImage* 202:116137 — MRtrix3
- Smith et al. (2015). *NeuroImage* 121:176-185 — SIFT2
- Efron & Tibshirani (1993). *An Introduction to the Bootstrap* — Bootstrap theory
- Zalesky, Fornito & Bullmore (2010). *NeuroImage* 53:1197-1207 — NBS
- Rubinov & Sporns (2010). *NeuroImage* 52:1059-1069 — Graph metrics
- Maier-Hein et al. (2017). *Nat Commun* 8:1349 — Tractography false positives
- Whitcher et al. (2008). *Hum Brain Mapp* 29:346-362 — Wild bootstrap for dMRI
- Jeurissen et al. (2011). *Hum Brain Mapp* 32:461-479 — Residual bootstrap for CSD
- Serrano, Boguñá & Vespignani (2009). *PNAS* 106:6483-6488 — Disparity filter
- Shen et al. (2017). *Nat Protoc* 12:506-518 — CPM

## License

MIT License. See [LICENSE](LICENSE) for details.

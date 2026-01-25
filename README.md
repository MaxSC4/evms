# EVMS: Volumetric Inversion of Geological Radioactivity

This project implements a Python pipeline for inverting volumetric radioactivity fields S(r) from surface gamma measurements, using sparse linear algebra and regularization for layered geology and finite fracture barriers.

## Scientific Overview

### Forward model

The forward model discretizes the volumetric integral relating the subsurface
radioactivity field $S(\mathbf{r})$ to surface measurements $M(\mathbf{x})$:

$$M(\mathbf{x})=\int_{\Omega}S(\mathbf{r})G(\mathbf{x}\mathbf{r})\\exp\!\big(-\mu\|\mathbf{x}-\mathbf{r}\|\big)\\mathrm{d}V$$

After spatial discretization on a 3D voxel grid, the forward problem can be
written in matrix form:

```math
\mathbf{M} \approx \mathbf{A}\mathbf{S} + \boldsymbol{\varepsilon}
```

The matrix \(\mathbf{A}\) is sparse due to truncation of interactions beyond
a maximum radius \(R_{\max}\).

---

### Inversion

The inverse problem is formulated as a regularized least-squares optimization:

```math
\min_{\mathbf{S}}
\ \|\mathbf{A}\mathbf{S} - \mathbf{M}\|_2^2
\ +\ \lambda\,\mathcal{R}(\mathbf{S})
```

where $\mathcal{R}(\mathbf{S})$ promotes spatial smoothing within geological
layers and reduced continuity across fracture surfaces.



## Installation

Create conda environment:
```bash
conda env create -f environment.yml
conda activate evms-env
pip install -e .
```

## Running the Streamlit App

```bash
streamlit run streamlit_app.py
```

Upload measurements CSV, grid NPY or OBJ (for voxelization, up to 500 MB), optional fractures JSON, set parameters, and run inversion.

Note: Large OBJ files may take time to voxelize; ensure sufficient RAM.

## Running Tests

```bash
pytest
```

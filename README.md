# EVMS — Volumetric Inversion of Geological Radioactivity

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)

EVMS is a scientific Python framework to reconstruct a 3D subsurface source-intensity field from surface gamma measurements. It combines a sparse forward model, geological regularization, and an interactive Streamlit interface for reproducible inversion workflows.

## Why EVMS

- Sparse, scalable inversion on voxel grids (`scipy.sparse`, iterative solvers).
- Physics-informed forward model with attenuation and distance kernel.
- Geological priors through layer-aware smoothing and finite fracture barriers.
- Publication-ready outputs: volumetric field export and textured mesh export.
- Interactive analysis, diagnostics, and trust reporting in Streamlit.

## Scientific Model (Concise)

Forward model:

$$
M(\mathbf{x}) = \int_V S(\mathbf{r})\,G(\mathbf{x},\mathbf{r})\,e^{-\mu\|\mathbf{x}-\mathbf{r}\|}\,dV + \varepsilon,
\qquad
G(\mathbf{x},\mathbf{r}) = \frac{1}{\|\mathbf{x}-\mathbf{r}\|^2 + \epsilon}
$$

Voxel discretization:

$$
\mathbf{M} \approx \mathbf{A}\mathbf{S} + \boldsymbol{\varepsilon}
$$

Regularized inversion:

$$
\hat{\mathbf{S}} = \arg\min_{\mathbf{S}}\;\|\mathbf{A}\mathbf{S}-\mathbf{M}\|_2^2 + \lambda\|\mathbf{L}\mathbf{S}\|_2^2
$$

where $\mathbf{L}$ is a graph-difference regularizer weighted by layer consistency and fracture crossing penalties.

## Installation

```bash
conda env create -f environment.yml
conda activate evms-env
pip install -e .
```

## Run the App

```bash
streamlit run streamlit_app.py
```

The app provides:
- Inversion workspace
- Method page (equations and workflow)
- Credits page

## Input Data

- **Measurements CSV**: `x,y,z,value`
- **Grid**: `.npy` mask or `.obj` mesh (voxelized in-app)
- **Fractures JSON**: optional finite rectangular fracture patches
- **Calibration CSV**: optional `x,y,z,cps/s`

## Validation and Testing

```bash
pytest
```

The test suite covers geometry, forward operator behavior, regularization, inversion, calibration, and diagnostics.

## Project Website

Documentation and project updates: [maxsc4.github.io](https://maxsc4.github.io)

## Contact

Maxime SOARES CORREIA — [soarescorreia@ipgp.fr](mailto:soarescorreia@ipgp.fr)

## Citation

A Zenodo DOI release is planned. Citation metadata will be added here once the DOI is minted.

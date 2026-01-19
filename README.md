# EVMS: Volumetric Inversion of Geological Radioactivity

This project implements a Python pipeline for inverting volumetric radioactivity fields S(r) from surface gamma measurements, using sparse linear algebra and regularization for layered geology and finite fracture barriers.

## Scientific Overview

The forward model discretizes the integral M(x) = ∫ S(r) G(x,r) exp(-μ ||x-r||) dV into M ≈ A S + ε, with A sparse via truncation to R_max.

Inversion solves min ||A S - M||^2 + λ R(S), where R promotes smoothing within layers and barriers across fractures.

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
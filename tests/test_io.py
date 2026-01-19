"""
Tests for io.py
"""

import os
import tempfile

import numpy as np
import trimesh

from evms.io import load_obj_as_grid


def test_load_obj_as_grid_aligns_origin():
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh.apply_translation([5.0, 5.0, 5.0])
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as f:
        obj_path = f.name
    try:
        mesh.export(obj_path)
        origin = (4.0, 4.0, 4.0)
        spacing = (1.0, 1.0, 1.0)
        dims = (4, 4, 4)
        grid = load_obj_as_grid(obj_path, origin, spacing, dims)
        assert grid.n_voxels > 0
        assert grid.mask[0, 0, 0]
        active = np.argwhere(grid.mask)
        mesh_bounds = mesh.bounds
        min_idx = np.floor((mesh_bounds[0] - np.array(origin)) / np.array(spacing)).astype(int)
        max_idx = np.floor((mesh_bounds[1] - np.array(origin)) / np.array(spacing)).astype(int)
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array(dims) - 1)
        assert (active >= min_idx).all()
        assert (active <= max_idx).all()
    finally:
        os.unlink(obj_path)

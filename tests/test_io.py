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
        assert grid.mask[1, 1, 1]
        assert not grid.mask[2, 2, 2]
    finally:
        os.unlink(obj_path)

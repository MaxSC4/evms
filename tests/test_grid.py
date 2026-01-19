"""
Tests for grid.py
"""

import numpy as np
import pytest
from evms.grid import VoxelGrid


def test_voxel_centers():
    grid = VoxelGrid((0,0,0), (1,1,1), (2,2,2))
    centers = grid.voxel_centers()
    expected = np.array([
        [0.5,0.5,0.5], [0.5,0.5,1.5], [0.5,1.5,0.5], [0.5,1.5,1.5],
        [1.5,0.5,0.5], [1.5,0.5,1.5], [1.5,1.5,0.5], [1.5,1.5,1.5]
    ])
    np.testing.assert_array_almost_equal(centers, expected)


def test_neighbor_edges():
    grid = VoxelGrid((0,0,0), (1,1,1), (2,2,2))
    edges = grid.neighbor_edges()
    assert len(edges) > 0  # Basic check
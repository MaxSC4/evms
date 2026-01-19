"""
Tests for regularization.py
"""

import numpy as np
from evms.grid import VoxelGrid
from evms.regularization import build_regularization_matrix


def test_build_reg():
    grid = VoxelGrid((0,0,0), (1,1,1), (2,2,2))
    labels = np.zeros(8)
    L = build_regularization_matrix(grid, labels)
    assert L.shape == (len(grid.neighbor_edges()), 8)
"""
Regularization matrix for layered geology and fractures.

Builds L such that ||L S||^2 = R(S).
"""

import numpy as np
from scipy import sparse
from typing import List, Optional
from .grid import VoxelGrid
from .fractures import Fracture, fracture_crossing_weight


def build_regularization_matrix(
    grid: VoxelGrid,
    layer_labels: np.ndarray,
    fractures: Optional[List[Fracture]] = None
) -> sparse.csr_matrix:
    """
    Build regularization matrix L.

    R(S) = Σ_{j~j'} w(j,j') (S_j - S_j')^2
    w = w_layer + w_fract
    w_layer = 1 if same layer else 0
    w_fract = Σ αℓ g(Dℓ) if crosses fracture ℓ

    L is such that rows correspond to edges, with sqrt(w) for each difference.

    Args:
        grid: VoxelGrid.
        layer_labels: Array of shape (n_voxels,) with layer indices.
        fractures: List of Fracture objects.

    Returns:
        Sparse CSR matrix L of shape (n_edges, n_voxels).
    """
    edges = grid.neighbor_edges()
    n_edges = len(edges)
    n_voxels = grid.n_voxels
    centers = grid.voxel_centers()

    rows, cols, data = [], [], []

    for e_idx, (j, jp) in enumerate(edges):
        w_layer = 1.0 if layer_labels[j] == layer_labels[jp] else 0.0
        w_fract = 0.0
        if fractures:
            r_j = centers[j]
            r_jp = centers[jp]
            for frac in fractures:
                w_fract += fracture_crossing_weight((r_j, r_jp), frac)
        w_total = w_layer + w_fract
        if w_total > 0:
            sqrt_w = np.sqrt(w_total)
            # For (S_j - S_jp)^2, L has +sqrt_w for j, -sqrt_w for jp
            rows.extend([e_idx, e_idx])
            cols.extend([j, jp])
            data.extend([sqrt_w, -sqrt_w])

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_edges, n_voxels))

"""
Forward operator for radioactivity measurements.

Builds sparse matrix A such that M ≈ A S + ε.
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from .grid import VoxelGrid


def build_forward_operator(
    grid: VoxelGrid,
    measurement_points: np.ndarray,
    mu: float,
    R_max: float,
    eps: float = 1e-6
) -> sparse.csr_matrix:
    """
    Build forward operator A.

    A_ij = G(x_i, r_j) * exp(-mu * ||x_i - r_j||) * ΔV_j if ||x_i - r_j|| < R_max, else 0.

    Simplifications:
    - mu constant
    - G(x,r) = 1 / (||x-r||^2 + eps)
    - Truncated to R_max for sparsity

    Args:
        grid: VoxelGrid with active voxels.
        measurement_points: Array of shape (n_points, 3) with (x,y,z).
        mu: Attenuation coefficient (1/m).
        R_max: Max influence radius (m).
        eps: Small value to avoid division by zero.

    Returns:
        Sparse CSR matrix of shape (n_points, n_voxels).
    """
    centers = grid.voxel_centers()
    n_points, _ = measurement_points.shape
    n_voxels = grid.n_voxels
    dx, dy, dz = grid.spacing
    volume = dx * dy * dz

    rows, cols, data = [], [], []
    tree = cKDTree(centers)

    for i in range(n_points):
        x = measurement_points[i]
        neighbor_idx = tree.query_ball_point(x, r=R_max)
        if not neighbor_idx:
            continue
        neighbor_idx = np.asarray(neighbor_idx, dtype=int)
        diffs = centers[neighbor_idx] - x
        dists = np.linalg.norm(diffs, axis=1)
        g = 1 / (dists**2 + eps)
        atten = np.exp(-mu * dists)
        a_ij = g * atten * volume
        rows.extend([i] * len(neighbor_idx))
        cols.extend(neighbor_idx.tolist())
        data.extend(a_ij.tolist())

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_points, n_voxels))

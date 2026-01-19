"""
Forward operator for radioactivity measurements.

Builds sparse matrix A such that M ≈ A S + ε.
"""

import numpy as np
from scipy import sparse
from typing import Union
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

    for i in range(n_points):
        x = measurement_points[i]
        for j in range(n_voxels):
            r = centers[j]
            dist = np.linalg.norm(x - r)
            if dist < R_max:
                g = 1 / (dist**2 + eps)
                atten = np.exp(-mu * dist)
                a_ij = g * atten * volume
                rows.append(i)
                cols.append(j)
                data.append(a_ij)

    return sparse.csr_matrix((data, (rows, cols)), shape=(n_points, n_voxels))
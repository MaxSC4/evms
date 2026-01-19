"""
Grid utilities for voxel discretization.

Provides VoxelGrid dataclass for 3D voxel grids with masking and indexing.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VoxelGrid:
    """
    Represents a 3D voxel grid.

    Attributes:
        origin: (x0, y0, z0) bottom-left-front corner in meters.
        spacing: (dx, dy, dz) voxel sizes in meters.
        dims: (nx, ny, nz) number of voxels in each direction.
        mask: Optional boolean array of shape (nx, ny, nz), True for active voxels.
    """
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    dims: Tuple[int, int, int]
    mask: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.ones(self.dims, dtype=bool)
        else:
            assert self.mask.shape == self.dims, "Mask shape must match dims"

    @property
    def n_voxels(self) -> int:
        """Total number of active voxels."""
        return np.sum(self.mask)

    def voxel_centers(self) -> np.ndarray:
        """
        Compute centers of active voxels.

        Returns:
            Array of shape (n_voxels, 3) with (x, y, z) coordinates.
        """
        x0, y0, z0 = self.origin
        dx, dy, dz = self.spacing
        nx, ny, nz = self.dims

        x = x0 + (np.arange(nx) + 0.5) * dx
        y = y0 + (np.arange(ny) + 0.5) * dy
        z = z0 + (np.arange(nz) + 0.5) * dz

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        centers = np.stack([X, Y, Z], axis=-1)
        centers = centers[self.mask]
        return centers

    def flat_to_ijk(self, flat_idx: int) -> Tuple[int, int, int]:
        """Convert flat index to (i,j,k) for active voxels."""
        active_indices = np.where(self.mask.ravel())[0]
        ijk_flat = active_indices[flat_idx]
        i, j, k = np.unravel_index(ijk_flat, self.dims)
        return i, j, k

    def ijk_to_flat(self, i: int, j: int, k: int) -> int:
        """Convert (i,j,k) to flat index if active."""
        if not self.mask[i, j, k]:
            raise ValueError("Voxel not active")
        flat = np.ravel_multi_index((i, j, k), self.dims)
        active_indices = np.where(self.mask.ravel())[0]
        return np.searchsorted(active_indices, flat)

    def neighbor_edges(self) -> np.ndarray:
        """
        Generate 6-neighbor edges for active voxels.

        Returns:
            Array of shape (n_edges, 2) with flat indices of connected voxels.
        """
        nx, ny, nz = self.dims
        edges = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if not self.mask[i, j, k]:
                        continue
                    flat = self.ijk_to_flat(i, j, k)
                    # 6 directions
                    for di, dj, dk in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        ni, nj, nk = i+di, j+dj, k+dk
                        if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz and self.mask[ni, nj, nk]:
                            nflat = self.ijk_to_flat(ni, nj, nk)
                            edges.append((flat, nflat))
        return np.array(edges)
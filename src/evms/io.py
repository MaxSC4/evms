"""
IO utilities for loading/saving data.
"""

import numpy as np
import json
import trimesh
from typing import List, Tuple
from .grid import VoxelGrid
from .fractures import Fracture
from scipy.spatial import cKDTree


def load_measurements(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load measurements from CSV: x,y,z,value (with or without header)
    """
    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()
    # Check if first line is header (contains letters)
    skiprows = 1 if any(c.isalpha() for c in first_line) else 0
    data = np.loadtxt(csv_path, delimiter=',', skiprows=skiprows)
    return data[:, :3], data[:, 3]


def load_grid(npy_path: str, origin: Tuple[float,float,float], spacing: Tuple[float,float,float]) -> VoxelGrid:
    """
    Load mask from .npy and create VoxelGrid.
    """
    mask = np.load(npy_path)
    dims = mask.shape
    return VoxelGrid(origin, spacing, dims, mask)


def load_fractures(json_path: str) -> List[Fracture]:
    """
    Load fractures from JSON list.
    """
    with open(json_path) as f:
        data = json.load(f)
    fractures = []
    for d in data:
        frac = Fracture(
            center=d['center'],
            normal=d['normal'],
            u=d['u'],
            v=d['v'],
            length=d['length'],
            width=d['width'],
            rho0=d.get('rho0', 1.0),
            rho1=d.get('rho1', 0.5),
            rho_min=d.get('rho_min', 0.1),
            alpha=d.get('alpha', 1.0)
        )
        fractures.append(frac)
    return fractures


def load_obj_as_grid(obj_path: str, origin: Tuple[float,float,float], spacing: Tuple[float,float,float], dims: Tuple[int,int,int]) -> VoxelGrid:
    """
    Load .obj mesh and voxelize to create mask for VoxelGrid.

    Assumes the mesh is the surface, and voxels inside/near surface are active.
    Uses trimesh voxelization with isotropic pitch.
    """
    mesh = trimesh.load(obj_path)
    spacing_arr = np.array(spacing, dtype=float)
    if not np.allclose(spacing_arr, spacing_arr[0]):
        raise ValueError("load_obj_as_grid requires isotropic spacing to match voxelization pitch")
    pitch = float(spacing_arr[0])
    voxel_grid = mesh.voxelized(pitch=pitch)
    points = voxel_grid.points

    mask = np.zeros(dims, dtype=bool)
    if points.size == 0:
        return VoxelGrid(origin, spacing, dims, mask)

    origin_arr = np.array(origin, dtype=float)
    rel = (points - origin_arr) / spacing_arr
    idx = np.floor(rel).astype(int)
    valid = (
        (idx[:, 0] >= 0) & (idx[:, 0] < dims[0]) &
        (idx[:, 1] >= 0) & (idx[:, 1] < dims[1]) &
        (idx[:, 2] >= 0) & (idx[:, 2] < dims[2])
    )
    idx = idx[valid]
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return VoxelGrid(origin, spacing, dims, mask)


def save_grid(grid: VoxelGrid, S: np.ndarray, npy_path: str):
    """
    Save S as 3D array (masked).
    """
    full_S = np.full(grid.dims, np.nan)
    flat_idx = 0
    for i in range(grid.dims[0]):
        for j in range(grid.dims[1]):
            for k in range(grid.dims[2]):
                if grid.mask[i,j,k]:
                    full_S[i,j,k] = S[flat_idx]
                    flat_idx += 1
    np.save(npy_path, full_S)


def apply_radioactivity_to_mesh(mesh: trimesh.Trimesh, grid: VoxelGrid, S_hat: np.ndarray) -> trimesh.Trimesh:
    """
    Apply radioactivity S_hat to mesh vertices as colors.

    For each vertex, find the nearest active voxel and assign its S_hat as color.
    Colors are normalized to 0-255 for RGB (grayscale).
    """
    centers = grid.voxel_centers()
    tree = cKDTree(centers)
    _, indices = tree.query(mesh.vertices)
    # Get S_hat for each vertex
    vertex_S = S_hat[indices]
    # Normalize to 0-1
    if vertex_S.max() > vertex_S.min():
        vertex_S_norm = (vertex_S - vertex_S.min()) / (vertex_S.max() - vertex_S.min())
    else:
        vertex_S_norm = np.zeros_like(vertex_S)
    # To RGB grayscale
    colors = (vertex_S_norm * 255).astype(np.uint8)
    colors_rgb = np.column_stack([colors, colors, colors, np.full_like(colors, 255)])  # RGBA
    mesh.visual.vertex_colors = colors_rgb
    return mesh

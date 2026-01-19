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
    # Compute voxel size as min spacing
    pitch = min(spacing)
    # Voxelize
    voxel_grid = mesh.voxelized(pitch=pitch)
    # Get the matrix
    matrix = voxel_grid.matrix
    # But our dims may not match; we need to resample or assume dims match the voxelized grid.
    # For simplicity, assume the voxelized grid matches dims.
    # But dims are given, so perhaps crop or pad.
    # To keep simple, use the voxelized matrix as mask, but adjust dims.
    # Perhaps better: define the grid bounds based on mesh bounds.
    bounds = mesh.bounds
    # origin is given, spacing given, dims given.
    # Compute expected bounds
    expected_bounds = np.array([
        [origin[0], origin[1], origin[2]],
        [origin[0] + dims[0]*spacing[0], origin[1] + dims[1]*spacing[1], origin[2] + dims[2]*spacing[2]]
    ])
    # Voxelize with the given pitch, but translate to origin.
    # Trimesh voxelize centers at 0, need to translate.
    # This is tricky. For simplicity, voxelize the mesh, then create mask where voxels are filled.
    # Assume dims is large enough, and set mask to True where voxelized has True.
    # But shapes may not match.
    # Let's assume pitch = spacing[0] assuming isotropic, and dims match.
    pitch = spacing[0]
    voxel_grid = mesh.voxelized(pitch=pitch)
    matrix = voxel_grid.matrix
    # The matrix shape is the dims of the voxel grid.
    # If it doesn't match dims, we need to handle.
    # For now, assume it does, or crop/pad.
    # To make it work, let's create mask of shape dims, and set to True where matrix has True, assuming alignment.
    mask = np.zeros(dims, dtype=bool)
    vshape = matrix.shape
    min_shape = min(dims[0], vshape[0]), min(dims[1], vshape[1]), min(dims[2], vshape[2])
    mask[:min_shape[0], :min_shape[1], :min_shape[2]] = matrix[:min_shape[0], :min_shape[1], :min_shape[2]]
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
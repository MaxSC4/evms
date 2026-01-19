"""
IO utilities for loading/saving data.
"""

import numpy as np
import json
import os
import trimesh
from typing import List, Tuple
from .grid import VoxelGrid
from .fractures import Fracture
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


def _normalize_bounds(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)], dtype=float)
    size = bounds[1] - bounds[0]
    size[size == 0.0] = 1.0
    return bounds[0], bounds[1], size


def _box_project_uv(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray) -> np.ndarray:
    """
    Compute per-face UVs using a box projection atlas (3x2 tiles).
    """
    vmin, vmax, vsize = _normalize_bounds(vertices)
    tile_w = 1.0 / 3.0
    tile_h = 1.0 / 2.0
    tile_map = {
        (0, 1): (0, 0),   # +X
        (0, -1): (1, 0),  # -X
        (1, 1): (2, 0),   # +Y
        (1, -1): (0, 1),  # -Y
        (2, 1): (1, 1),   # +Z
        (2, -1): (2, 1),  # -Z
    }

    face_uvs = np.zeros((faces.shape[0], 3, 2), dtype=float)
    for f_idx, face in enumerate(faces):
        n = face_normals[f_idx]
        axis = int(np.argmax(np.abs(n)))
        sign = 1 if n[axis] >= 0 else -1
        tx, ty = tile_map[(axis, sign)]

        verts = vertices[face]
        normed = (verts - vmin) / vsize
        if axis == 0:
            u = normed[:, 2]
            v = normed[:, 1]
        elif axis == 1:
            u = normed[:, 0]
            v = normed[:, 2]
        else:
            u = normed[:, 0]
            v = normed[:, 1]
        if sign < 0:
            u = 1.0 - u

        face_uvs[f_idx, :, 0] = (tx * tile_w) + u * tile_w
        face_uvs[f_idx, :, 1] = (ty * tile_h) + v * tile_h

    return face_uvs


def _bake_texture(
    faces: np.ndarray,
    uvs: np.ndarray,
    values: np.ndarray,
    image_size: int
) -> np.ndarray:
    """
    Rasterize per-face UVs and scalar values into a texture image.
    """
    img = np.zeros((image_size, image_size), dtype=float)
    weight = np.zeros((image_size, image_size), dtype=float)

    vmin = float(values.min())
    vmax = float(values.max())
    scale = vmax - vmin if vmax > vmin else 1.0
    vals = (values - vmin) / scale

    for f_idx, face in enumerate(faces):
        uv = uvs[f_idx]
        pix = np.empty_like(uv)
        pix[:, 0] = uv[:, 0] * (image_size - 1)
        pix[:, 1] = (1.0 - uv[:, 1]) * (image_size - 1)
        xmin = int(np.floor(pix[:, 0].min()))
        xmax = int(np.ceil(pix[:, 0].max()))
        ymin = int(np.floor(pix[:, 1].min()))
        ymax = int(np.ceil(pix[:, 1].max()))
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, image_size - 1)
        ymax = min(ymax, image_size - 1)

        x0, y0 = pix[0]
        x1, y1 = pix[1]
        x2, y2 = pix[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if denom == 0:
            continue

        v0 = vals[face[0]]
        v1 = vals[face[1]]
        v2 = vals[face[2]]

        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                w0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
                w1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
                w2 = 1.0 - w0 - w1
                if (w0 >= 0) and (w1 >= 0) and (w2 >= 0):
                    val = w0 * v0 + w1 * v1 + w2 * v2
                    img[y, x] += val
                    weight[y, x] += 1.0

    mask = weight > 0
    img[mask] /= weight[mask]
    img = np.clip(img, 0.0, 1.0)
    rgb = (img * 255.0).astype(np.uint8)
    return np.stack([rgb, rgb, rgb], axis=-1)


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


def apply_radioactivity_texture(
    mesh: trimesh.Trimesh,
    grid: VoxelGrid,
    S_hat: np.ndarray,
    image_size: int = 1024
) -> trimesh.Trimesh:
    """
    Bake radioactivity into a texture image with box-projected UVs.
    """
    centers = grid.voxel_centers()
    tree = cKDTree(centers)
    _, indices = tree.query(mesh.vertices)
    vertex_S = S_hat[indices]

    faces = mesh.faces
    face_normals = mesh.face_normals
    face_uvs = _box_project_uv(mesh.vertices, faces, face_normals)

    flat_vertices = mesh.vertices[faces].reshape(-1, 3)
    flat_uvs = face_uvs.reshape(-1, 2)
    flat_faces = np.arange(flat_vertices.shape[0]).reshape(-1, 3)

    texture = _bake_texture(faces, face_uvs, vertex_S, image_size)
    textured_mesh = trimesh.Trimesh(vertices=flat_vertices, faces=flat_faces, process=False)
    textured_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=flat_uvs, image=texture)
    return textured_mesh


def export_textured_obj(
    mesh: trimesh.Trimesh,
    obj_path: str,
    texture_image: np.ndarray,
    material_name: str = "material0"
) -> Tuple[str, str, str]:
    """
    Export OBJ with MTL and PNG texture in a viewer-agnostic way.
    """
    base_dir = os.path.dirname(obj_path)
    base_name = os.path.splitext(os.path.basename(obj_path))[0]
    mtl_name = f"{base_name}.mtl"
    tex_name = f"{base_name}.png"
    mtl_path = os.path.join(base_dir, mtl_name)
    tex_path = os.path.join(base_dir, tex_name)

    plt.imsave(tex_path, texture_image)

    with open(mtl_path, "w") as f:
        f.write(f"newmtl {material_name}\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.000\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {tex_name}\n")

    vertices = mesh.vertices
    faces = mesh.faces
    uvs = mesh.visual.uv

    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        f.write(f"usemtl {material_name}\n")
        for face in faces:
            a, b, c = face + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    return obj_path, mtl_path, tex_path

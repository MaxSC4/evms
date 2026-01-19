"""
Streamlit UI for EVMS inversion.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import trimesh
import plotly.graph_objects as go
from evms import (
    VoxelGrid, build_forward_operator, build_regularization_matrix,
    solve_tikhonov, select_lambda, load_measurements, load_grid,
    load_fractures, save_grid, load_obj_as_grid, apply_radioactivity_to_mesh, apply_radioactivity_texture, export_textured_obj, compute_residuals
)

st.title("EVMS: Radioactivity Inversion")

# Sidebar params
st.sidebar.header("Parameters")
mu = st.sidebar.slider("Attenuation mu", 0.0, 0.1, 0.01)
R_max = st.sidebar.slider("R_max", 1.0, 20.0, 5.0)
lam_manual = st.sidebar.slider("Lambda (manual)", 1e-3, 1e1, 1.0, format="%.3f")
use_auto_lam = st.sidebar.checkbox("Auto select lambda", value=True)

# File uploads
st.header("Data Upload")
meas_file = st.file_uploader("Measurements CSV (x,y,z,value)", type="csv")
grid_file = st.file_uploader("Grid NPY or OBJ", type=["npy", "obj"])
frac_file = st.file_uploader("Fractures JSON (optional)", type="json")

# Compute defaults for grid if OBJ
default_origin = (0.0, 0.0, 0.0)
default_spacing = 1.0
default_dims = (10, 10, 10)
if grid_file and grid_file.name.endswith('.obj'):
    with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
        f.write(grid_file.getvalue())
        obj_temp = f.name
    mesh = trimesh.load(obj_temp)
    bounds = mesh.bounds
    default_origin = tuple(bounds[0])
    size = bounds[1] - bounds[0]
    default_spacing = min(size) / 10  # isotropic, based on smallest dimension
    default_dims = tuple(int(np.ceil(s / default_spacing)) for s in size)
    os.unlink(obj_temp)

# Grid parameters
st.subheader("Grid Parameters")
origin_x = st.number_input("Origin X", value=default_origin[0])
origin_y = st.number_input("Origin Y", value=default_origin[1])
origin_z = st.number_input("Origin Z", value=default_origin[2])
spacing = st.number_input("Spacing (isotropic)", value=default_spacing)
dims_x = st.number_input("Dims X", value=default_dims[0], min_value=1)
dims_y = st.number_input("Dims Y", value=default_dims[1], min_value=1)
dims_z = st.number_input("Dims Z", value=default_dims[2], min_value=1)

if meas_file and grid_file:
    # Save uploaded files to temp
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(meas_file.getvalue().decode('utf-8'))
        meas_temp = f.name
    points, M = load_measurements(meas_temp)
    os.unlink(meas_temp)  # clean up

    origin = (origin_x, origin_y, origin_z)
    spacing = (spacing, spacing, spacing)  # isotropic
    dims = (int(dims_x), int(dims_y), int(dims_z))
    if grid_file.name.endswith('.npy'):
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            f.write(grid_file.getvalue())
            grid_temp = f.name
        mask = np.load(grid_temp)
        grid = VoxelGrid(origin, spacing, dims, mask)
        os.unlink(grid_temp)
    elif grid_file.name.endswith('.obj'):
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            f.write(grid_file.getvalue())
            obj_temp = f.name
        grid = load_obj_as_grid(obj_temp, origin, spacing, dims)
        os.unlink(obj_temp)

    fractures = []
    if frac_file:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(frac_file.getvalue().decode('utf-8'))
            frac_temp = f.name
        fractures = load_fractures(frac_temp)
        os.unlink(frac_temp)

    # Assume layers all 0
    layer_labels = np.zeros(grid.n_voxels, dtype=int)

    # Build operators
    A = build_forward_operator(grid, points, mu, R_max)
    L = build_regularization_matrix(grid, layer_labels, fractures)

    # Select lambda
    if use_auto_lam:
        lambda_grid = np.logspace(-3, 1, 10)
        lam, _ = select_lambda(A, M, L, lambda_grid)
    else:
        lam = lam_manual

    # Solve
    S_hat = solve_tikhonov(A, M, L, lam)

    # Display
    st.write(f"Best lambda: {lam}")
    res = compute_residuals(A, M, S_hat)
    st.write(f"Residual norm: {np.linalg.norm(res)}")

    st.subheader("Résultats de l'inversion")
    st.write("""
    **S_hat** est le champ volumique de radioactivité estimé S(r), reconstruit à partir des mesures de surface.
    C'est la solution du problème d'optimisation : min ||A S - M||² + λ ||L S||²
    - A : opérateur forward (modèle physique)
    - M : mesures de surface
    - L : matrice de régularisation (lissage intra-couche + barrières fractures)
    - λ : paramètre de régularisation sélectionné automatiquement
    """)

    st.write(f"Nombre de voxels actifs : {grid.n_voxels}")
    st.write(f"Radioactivité estimée - Min: {S_hat.min():.2f}, Max: {S_hat.max():.2f}, Moyenne: {S_hat.mean():.2f}")

    # Histogram
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(S_hat, bins=20, alpha=0.7)
    ax_hist.set_title("Distribution de la radioactivité estimée")
    ax_hist.set_xlabel("Radioactivité (Bq/kg)")
    ax_hist.set_ylabel("Nombre de voxels")
    st.pyplot(fig_hist)

    # Plot slice
    full_S = np.full(dims, np.nan)
    flat_idx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if grid.mask[i,j,k]:
                    full_S[i,j,k] = S_hat[flat_idx]
                    flat_idx += 1
    slice_k = st.slider("Slice k", 0, dims[2]-1, min(5, dims[2]-1))
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(full_S[:, :, slice_k], origin='lower', cmap='viridis')
    ax.set_title(f"Reconstructed Radioactivity Slice (k={slice_k})")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Radioactivity (Bq/kg)")
    st.pyplot(fig)

    # 3D Plot
    st.subheader("Visualisation 3D")
    centers = grid.voxel_centers()
    fig_3d = go.Figure(data=go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=S_hat,
            colorscale='Viridis',
            colorbar=dict(title="Radioactivity (Bq/kg)"),
            showscale=True
        )
    ))
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Champ de radioactivité reconstitué (3D)"
    )
    st.plotly_chart(fig_3d)

    # Export mesh with baked texture (OBJ + MTL + PNG)
    if st.button("Exporter mesh avec texture radioactivité"):
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            f.write(grid_file.getvalue())
            obj_temp = f.name
        mesh = trimesh.load(obj_temp)
        os.unlink(obj_temp)
        try:
            textured_mesh = apply_radioactivity_texture(mesh, grid, S_hat, image_size=1024)
            with tempfile.TemporaryDirectory() as tmpdir:
                obj_path = os.path.join(tmpdir, "radioactivity_mesh.obj")
                export_textured_obj(textured_mesh, obj_path, textured_mesh.visual.material.image)
                st.success("OBJ exporté avec texture (OBJ + MTL + PNG)")
                for ext in ("obj", "mtl", "png"):
                    matches = [p for p in os.listdir(tmpdir) if p.lower().endswith(f".{ext}")]
                    for name in matches:
                        path = os.path.join(tmpdir, name)
                        with open(path, "rb") as f:
                            st.download_button(f"Télécharger {name}", f, file_name=name)
        except Exception as e:
            st.error(f"Erreur export: {e}")

    # Export
    if st.button("Export S_hat"):
        save_grid(grid, S_hat, "S_hat.npy")
        st.success("Saved S_hat.npy")

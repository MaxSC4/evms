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
    solve_tikhonov, select_lambda, select_forward_params, load_measurements, load_grid,
    load_fractures, save_grid, load_obj_as_grid, apply_radioactivity_to_mesh, apply_radioactivity_texture,
    export_textured_obj, compute_residuals, fit_calibration_from_points, apply_calibration,
    compute_data_misfit_norm, compute_regularization_norm, compute_holdout_error
)

st.title("EVMS: Radioactivity Inversion")

# Sidebar params
st.sidebar.header("Parameters")
mu = st.sidebar.slider("Attenuation mu", 0.0, 0.1, 0.01)
R_max = st.sidebar.slider("R_max", 1.0, 20.0, 5.0)
lam_manual = st.sidebar.slider("Lambda (manual)", 1e-3, 1e1, 1.0, format="%.3f")
use_auto_lam = st.sidebar.checkbox("Auto select lambda", value=True)
auto_tune_forward = st.sidebar.checkbox("Auto tune mu and R_max", value=False)
if auto_tune_forward:
    st.sidebar.caption("Grid-search over forward-model parameters")
    mu_min = st.sidebar.slider("mu min", 0.0, 0.1, 0.0)
    mu_max = st.sidebar.slider("mu max", 0.0, 0.1, 0.05)
    mu_steps = st.sidebar.slider("mu steps", 2, 10, 4)
    rmax_min = st.sidebar.slider("R_max min", 1.0, 20.0, 2.0)
    rmax_max = st.sidebar.slider("R_max max", 1.0, 20.0, 10.0)
    rmax_steps = st.sidebar.slider("R_max steps", 2, 10, 4)
    tuning_objective_label = st.sidebar.selectbox(
        "Tuning objective",
        ["Residual norm", "Holdout RMSE"],
        index=0,
    )
else:
    tuning_objective_label = "Residual norm"

# File uploads
st.header("Data Upload")
meas_file = st.file_uploader("Measurements CSV (x,y,z,value)", type="csv")
grid_file = st.file_uploader("Grid NPY or OBJ", type=["npy", "obj"])
frac_file = st.file_uploader("Fractures JSON (optional)", type="json")
calib_file = st.file_uploader("Calibration CSV (x,y,z,cps/s) (optional)", type="csv")

# Optional calibration controls
use_calibration = st.sidebar.checkbox("Apply calibration", value=False)
fit_calib_offset = st.sidebar.checkbox("Calibration with offset", value=True)
use_holdout_diagnostics = st.sidebar.checkbox("Compute holdout diagnostics", value=False)
holdout_fraction = st.sidebar.slider("Holdout fraction", 0.1, 0.5, 0.2, 0.05)
holdout_seed = st.sidebar.number_input("Holdout random seed", min_value=0, value=42, step=1)

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
    L = build_regularization_matrix(grid, layer_labels, fractures)
    mu_used = mu
    rmax_used = R_max
    tuning_table = None

    if auto_tune_forward:
        mu_grid = np.linspace(mu_min, mu_max, mu_steps)
        rmax_grid = np.linspace(rmax_min, rmax_max, rmax_steps)
        if tuning_objective_label == "Holdout RMSE":
            tuning_objective = "holdout"
        else:
            tuning_objective = "residual"
        lambda_grid = np.logspace(-3, 1, 10) if use_auto_lam else None
        mu_used, rmax_used, lam, tuning_table = select_forward_params(
            grid=grid,
            measurement_points=points,
            M=M,
            L=L,
            mu_grid=mu_grid,
            rmax_grid=rmax_grid,
            lam=lam_manual if not use_auto_lam else None,
            lambda_grid=lambda_grid,
            objective=tuning_objective,
            holdout_fraction=float(holdout_fraction),
            random_state=int(holdout_seed),
        )
        A = build_forward_operator(grid, points, mu_used, rmax_used)
    else:
        A = build_forward_operator(grid, points, mu_used, rmax_used)
        if use_auto_lam:
            lambda_grid = np.logspace(-3, 1, 10)
            lam, _ = select_lambda(A, M, L, lambda_grid)
        else:
            lam = lam_manual

    # Solve
    S_hat = solve_tikhonov(A, M, L, lam)

    # Optional calibration on reconstructed source intensity
    S_display = S_hat
    value_unit = "relative units"
    calibration_model = None
    calibration_sampled = None
    calibration_targets = None
    if use_calibration:
        if calib_file is None:
            st.warning("Calibration enabled but no calibration CSV was provided. Using relative units.")
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                f.write(calib_file.getvalue().decode('utf-8'))
                calib_temp = f.name
            cal_points, cal_values = load_measurements(calib_temp)
            os.unlink(calib_temp)
            try:
                calibration_model, calibration_sampled = fit_calibration_from_points(
                    grid=grid,
                    s_hat=S_hat,
                    calibration_points=cal_points,
                    calibration_values=cal_values,
                    fit_offset=fit_calib_offset,
                )
                calibration_targets = cal_values
                S_display = apply_calibration(S_hat, calibration_model)
                value_unit = "cps/s"
            except ValueError as exc:
                st.error(f"Calibration error: {exc}")

    # Display
    st.write(f"Forward parameters: mu={mu_used:.4f}, R_max={rmax_used:.3f}")
    mu_warnings = []
    # Heuristic range for effective gamma attenuation in this simplified model.
    typical_mu_min = 1e-3
    typical_mu_max = 8e-2
    if mu_used <= 0.0:
        mu_warnings.append(
            "mu <= 0 means no attenuation, which is physically implausible for gamma transport in rock."
        )
    elif mu_used < typical_mu_min or mu_used > typical_mu_max:
        mu_warnings.append(
            f"mu={mu_used:.4f} is outside the typical range [{typical_mu_min:.4f}, {typical_mu_max:.4f}] 1/m; "
            "treat it as an effective fit parameter."
        )
    if auto_tune_forward:
        mu_span = max(mu_max - mu_min, 1e-12)
        mu_tol = 0.02 * mu_span
        if abs(mu_used - mu_min) <= mu_tol or abs(mu_used - mu_max) <= mu_tol:
            mu_warnings.append(
                "Tuned mu is at the edge of the search interval; expand/refine mu bounds to confirm identifiability."
            )
    for msg in mu_warnings:
        st.warning(msg)
    st.write(f"Best lambda: {lam}")
    if tuning_table is not None:
        st.subheader("Forward tuning results")
        st.dataframe(
            {
                "mu": tuning_table[:, 0],
                "R_max": tuning_table[:, 1],
                "lambda": tuning_table[:, 2],
                "score": tuning_table[:, 3],
            },
            use_container_width=True,
        )
    res = compute_residuals(A, M, S_hat)
    data_misfit_norm = compute_data_misfit_norm(A, M, S_hat)
    reg_norm = compute_regularization_norm(L, S_hat)
    holdout_diag = None
    if use_holdout_diagnostics:
        try:
            holdout_diag = compute_holdout_error(
                A=A,
                M=M,
                L=L,
                lam=lam,
                holdout_fraction=float(holdout_fraction),
                random_state=int(holdout_seed),
            )
        except ValueError as exc:
            st.warning(f"Holdout diagnostics unavailable: {exc}")

    st.write(f"Data misfit ||A S - M||: {data_misfit_norm:.4f}")
    st.write(f"Regularization ||L S||: {reg_norm:.4f}")
    if calibration_model is not None:
        st.write(
            "Calibration: "
            f"gain={calibration_model.gain:.6g}, offset={calibration_model.offset:.6g}, "
            f"R²={calibration_model.r2:.4f}, n={calibration_model.n_samples}"
        )

    st.subheader("Trust report")
    M_norm = float(np.linalg.norm(M))
    fit_ratio = data_misfit_norm / (M_norm + 1e-12)
    fit_score = 1.0 / (1.0 + fit_ratio)
    score_parts = [fit_score]
    holdout_txt = "Not computed"
    if holdout_diag is not None:
        holdout_scale = float(np.std(M) + 1e-12)
        holdout_ratio = holdout_diag.holdout_rmse / holdout_scale
        holdout_score = 1.0 / (1.0 + holdout_ratio)
        score_parts.append(holdout_score)
        holdout_txt = (
            f"RMSE={holdout_diag.holdout_rmse:.4f}, MAE={holdout_diag.holdout_mae:.4f}, "
            f"n_holdout={holdout_diag.holdout_size}"
        )
    trust_score = float(np.mean(score_parts))
    if trust_score >= 0.75:
        trust_level = "High"
        st.success(f"Trust level: {trust_level} (score={trust_score:.3f})")
    elif trust_score >= 0.45:
        trust_level = "Moderate"
        st.warning(f"Trust level: {trust_level} (score={trust_score:.3f})")
    else:
        trust_level = "Low"
        st.error(f"Trust level: {trust_level} (score={trust_score:.3f})")
    st.write(f"Fit ratio ||A S - M|| / ||M||: {fit_ratio:.4f}")
    st.write(f"Holdout diagnostics: {holdout_txt}")

    st.subheader("Residual map at measurements")
    fig_residual_3d = go.Figure(data=go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=res,
            colorscale='RdBu',
            reversescale=True,
            colorbar=dict(title="Residual (M - A S)"),
            showscale=True
        )
    ))
    fig_residual_3d.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title="Residual map at measurement points"
    )
    st.plotly_chart(fig_residual_3d)

    fig_res, ax_res = plt.subplots()
    ax_res.hist(res, bins=30, alpha=0.8)
    ax_res.set_title("Residual distribution")
    ax_res.set_xlabel("Residual (M - A S)")
    ax_res.set_ylabel("Count")
    st.pyplot(fig_res)

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
    st.write(
        f"Intensité de source estimée ({value_unit}) - "
        f"Min: {S_display.min():.2f}, Max: {S_display.max():.2f}, Moyenne: {S_display.mean():.2f}"
    )

    # Histogram
    fig_hist, ax_hist = plt.subplots()
    ax_hist.hist(S_display, bins=20, alpha=0.7)
    ax_hist.set_title("Distribution de l'intensité de source estimée")
    ax_hist.set_xlabel(f"Intensité de source ({value_unit})")
    ax_hist.set_ylabel("Nombre de voxels")
    st.pyplot(fig_hist)

    # Plot slice
    full_S = np.full(dims, np.nan)
    flat_idx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
                for k in range(dims[2]):
                    if grid.mask[i,j,k]:
                        full_S[i,j,k] = S_display[flat_idx]
                        flat_idx += 1
    slice_k = st.slider("Slice k", 0, dims[2]-1, min(5, dims[2]-1))
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(full_S[:, :, slice_k], origin='lower', cmap='RdYlGn_r')
    ax.set_title(f"Reconstructed Source Intensity Slice (k={slice_k})")
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Source intensity ({value_unit})")
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
            color=S_display,
            colorscale='RdYlGn',
            reversescale=True,
            colorbar=dict(title=f"Source intensity ({value_unit})"),
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
            textured_mesh = apply_radioactivity_texture(mesh, grid, S_display, image_size=1024)
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
        save_grid(grid, S_display, "S_hat.npy")
        st.success("Saved S_hat.npy")

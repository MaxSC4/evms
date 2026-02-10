"""Streamlit interface for EVMS inversion workflow."""

import os
import tempfile
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import trimesh

from evms import (
    VoxelGrid,
    apply_calibration,
    apply_radioactivity_texture,
    build_forward_operator,
    build_regularization_matrix,
    compute_data_misfit_norm,
    compute_holdout_error,
    compute_regularization_norm,
    compute_residuals,
    export_textured_obj,
    fit_calibration_from_points,
    load_fractures,
    load_measurements,
    load_obj_as_grid,
    save_grid,
    select_forward_params,
    select_lambda,
    solve_tikhonov,
)
from streamlit_ui import inject_global_css, render_page_header


inject_global_css()
render_page_header(
    "EVMS",
    "Volumetric inversion of geological radioactivity from surface gamma measurements",
)


def _uploaded_to_temp(uploaded_file: Any, suffix: str, text_mode: bool = False) -> str:
    mode = "w" if text_mode else "wb"
    with tempfile.NamedTemporaryFile(mode=mode, suffix=suffix, delete=False) as handle:
        if text_mode:
            handle.write(uploaded_file.getvalue().decode("utf-8"))
        else:
            handle.write(uploaded_file.getvalue())
        return handle.name


def _estimate_defaults_from_obj(uploaded_obj: Any) -> Tuple[Tuple[float, float, float], float, Tuple[int, int, int]]:
    obj_temp = _uploaded_to_temp(uploaded_obj, ".obj", text_mode=False)
    mesh = trimesh.load(obj_temp)
    os.unlink(obj_temp)
    bounds = mesh.bounds
    origin = tuple(bounds[0].astype(float))
    size = (bounds[1] - bounds[0]).astype(float)
    spacing = float(max(min(size) / 12.0, 1e-3))
    dims = tuple(int(np.ceil(axis / spacing)) for axis in size)
    return origin, spacing, dims


def _build_full_volume(grid: VoxelGrid, values: np.ndarray) -> np.ndarray:
    volume = np.full(grid.dims, np.nan, dtype=float)
    volume[grid.mask] = values
    return volume


def _compute_trust_score(
    M: np.ndarray,
    data_misfit_norm: float,
    holdout_diag: Optional[Any],
) -> Tuple[float, str, float, str]:
    m_norm = float(np.linalg.norm(M))
    fit_ratio = data_misfit_norm / (m_norm + 1e-12)
    fit_score = 1.0 / (1.0 + fit_ratio)
    score_parts = [fit_score]
    holdout_summary = "Not computed"

    if holdout_diag is not None:
        holdout_scale = float(np.std(M) + 1e-12)
        holdout_ratio = holdout_diag.holdout_rmse / holdout_scale
        holdout_score = 1.0 / (1.0 + holdout_ratio)
        score_parts.append(holdout_score)
        holdout_summary = (
            f"RMSE={holdout_diag.holdout_rmse:.4f}, MAE={holdout_diag.holdout_mae:.4f}, "
            f"n_holdout={holdout_diag.holdout_size}"
        )

    trust_score = float(np.mean(score_parts))
    if trust_score >= 0.75:
        trust_level = "High"
    elif trust_score >= 0.45:
        trust_level = "Moderate"
    else:
        trust_level = "Low"
    return trust_score, trust_level, fit_ratio, holdout_summary


def _run_inversion(payload: Dict[str, Any]) -> Dict[str, Any]:
    points, M = load_measurements(payload["measurement_csv"])

    spacing_scalar = float(payload["spacing"])
    spacing = (spacing_scalar, spacing_scalar, spacing_scalar)
    origin = tuple(float(v) for v in payload["origin"])

    if payload["grid_kind"] == "npy":
        mask = np.load(payload["grid_path"])
        grid = VoxelGrid(origin, spacing, mask.shape, mask)
    else:
        dims = tuple(int(v) for v in payload["dims"])
        grid = load_obj_as_grid(payload["grid_path"], origin, spacing, dims)

    fractures = []
    if payload["fracture_path"] is not None:
        fractures = load_fractures(payload["fracture_path"])

    layer_labels = np.zeros(grid.n_voxels, dtype=int)
    L = build_regularization_matrix(grid, layer_labels, fractures)

    mu_used = float(payload["mu"])
    rmax_used = float(payload["r_max"])
    tuning_table = None

    if payload["auto_tune_forward"]:
        mu_lo = min(payload["mu_min"], payload["mu_max"])
        mu_hi = max(payload["mu_min"], payload["mu_max"])
        r_lo = min(payload["rmax_min"], payload["rmax_max"])
        r_hi = max(payload["rmax_min"], payload["rmax_max"])

        mu_grid = np.linspace(mu_lo, mu_hi, int(payload["mu_steps"]))
        rmax_grid = np.linspace(r_lo, r_hi, int(payload["rmax_steps"]))
        lambda_grid = np.logspace(-3, 1, 10) if payload["use_auto_lam"] else None

        mu_used, rmax_used, lam, tuning_table = select_forward_params(
            grid=grid,
            measurement_points=points,
            M=M,
            L=L,
            mu_grid=mu_grid,
            rmax_grid=rmax_grid,
            lam=payload["lam_manual"] if not payload["use_auto_lam"] else None,
            lambda_grid=lambda_grid,
            objective=payload["tuning_objective"],
            holdout_fraction=float(payload["holdout_fraction"]),
            random_state=int(payload["holdout_seed"]),
        )
        A = build_forward_operator(grid, points, mu_used, rmax_used)
    else:
        A = build_forward_operator(grid, points, mu_used, rmax_used)
        if payload["use_auto_lam"]:
            lambda_grid = np.logspace(-3, 1, 10)
            lam, _ = select_lambda(A, M, L, lambda_grid)
        else:
            lam = float(payload["lam_manual"])

    S_hat = solve_tikhonov(A, M, L, lam)

    S_display = S_hat
    value_unit = "relative units"
    calibration_model = None

    if payload["use_calibration"] and payload["calibration_path"] is not None:
        cal_points, cal_values = load_measurements(payload["calibration_path"])
        calibration_model, _ = fit_calibration_from_points(
            grid=grid,
            s_hat=S_hat,
            calibration_points=cal_points,
            calibration_values=cal_values,
            fit_offset=payload["fit_calib_offset"],
        )
        S_display = apply_calibration(S_hat, calibration_model)
        value_unit = "cps/s"

    residuals = compute_residuals(A, M, S_hat)
    data_misfit_norm = compute_data_misfit_norm(A, M, S_hat)
    reg_norm = compute_regularization_norm(L, S_hat)

    holdout_diag = None
    if payload["use_holdout_diagnostics"]:
        holdout_diag = compute_holdout_error(
            A=A,
            M=M,
            L=L,
            lam=lam,
            holdout_fraction=float(payload["holdout_fraction"]),
            random_state=int(payload["holdout_seed"]),
        )

    trust_score, trust_level, fit_ratio, holdout_summary = _compute_trust_score(M, data_misfit_norm, holdout_diag)

    return {
        "grid": grid,
        "points": points,
        "M": M,
        "A": A,
        "L": L,
        "S_hat": S_hat,
        "S_display": S_display,
        "value_unit": value_unit,
        "residuals": residuals,
        "data_misfit_norm": data_misfit_norm,
        "reg_norm": reg_norm,
        "holdout_diag": holdout_diag,
        "fit_ratio": fit_ratio,
        "holdout_summary": holdout_summary,
        "trust_score": trust_score,
        "trust_level": trust_level,
        "calibration_model": calibration_model,
        "mu_used": mu_used,
        "rmax_used": rmax_used,
        "lam": lam,
        "tuning_table": tuning_table,
        "mu_search_bounds": (payload.get("mu_min"), payload.get("mu_max")),
        "mesh_bytes": payload.get("mesh_bytes"),
    }


def _render_mu_warnings(result: Dict[str, Any], auto_tune_forward: bool) -> None:
    mu_used = float(result["mu_used"])
    warnings = []

    typical_mu_min = 1e-3
    typical_mu_max = 8e-2

    if mu_used <= 0.0:
        warnings.append("`mu <= 0` implies no attenuation, which is physically implausible for gamma transport in rock.")
    elif mu_used < typical_mu_min or mu_used > typical_mu_max:
        warnings.append(
            f"`mu={mu_used:.4f}` is outside the typical effective range "
            f"`[{typical_mu_min:.4f}, {typical_mu_max:.4f}] 1/m`; interpret it as a compensating fit parameter."
        )

    if auto_tune_forward:
        bounds = result.get("mu_search_bounds")
        if bounds and bounds[0] is not None and bounds[1] is not None:
            mu_min = min(bounds[0], bounds[1])
            mu_max = max(bounds[0], bounds[1])
            mu_tol = 0.02 * max(mu_max - mu_min, 1e-12)
            if abs(mu_used - mu_min) <= mu_tol or abs(mu_used - mu_max) <= mu_tol:
                warnings.append("Tuned `mu` is near the search boundary; widen the interval to check identifiability.")

    for msg in warnings:
        st.warning(msg)


if "evms_result" not in st.session_state:
    st.session_state["evms_result"] = None

st.sidebar.header("Model Controls")
mu = st.sidebar.slider("Attenuation mu (1/m)", 0.0, 0.1, 0.01)
r_max = st.sidebar.slider("Influence radius R_max (m)", 1.0, 20.0, 5.0)
lam_manual = st.sidebar.slider("Lambda (manual)", 1e-3, 1e1, 1.0, format="%.3f")
use_auto_lam = st.sidebar.checkbox("Auto-select lambda (L-curve)", value=True)

auto_tune_forward = st.sidebar.checkbox("Auto-tune mu and R_max", value=False)
if auto_tune_forward:
    st.sidebar.caption("Grid-search over forward-model parameters")
    mu_min = st.sidebar.slider("mu min", 0.0, 0.1, 0.0)
    mu_max = st.sidebar.slider("mu max", 0.0, 0.1, 0.05)
    mu_steps = st.sidebar.slider("mu steps", 2, 10, 4)
    rmax_min = st.sidebar.slider("R_max min", 1.0, 20.0, 2.0)
    rmax_max = st.sidebar.slider("R_max max", 1.0, 20.0, 10.0)
    rmax_steps = st.sidebar.slider("R_max steps", 2, 10, 4)
    tuning_objective = st.sidebar.selectbox("Tuning objective", ["Residual norm", "Holdout RMSE"], index=0)
else:
    mu_min = mu_max = mu
    mu_steps = 2
    rmax_min = rmax_max = r_max
    rmax_steps = 2
    tuning_objective = "Residual norm"

st.sidebar.header("Diagnostics Controls")
use_holdout_diagnostics = st.sidebar.checkbox("Compute holdout diagnostics", value=False)
holdout_fraction = st.sidebar.slider("Holdout fraction", 0.1, 0.5, 0.2, 0.05)
holdout_seed = st.sidebar.number_input("Holdout random seed", min_value=0, value=42, step=1)

st.sidebar.header("Calibration Controls")
use_calibration = st.sidebar.checkbox("Apply calibration", value=False)
fit_calib_offset = st.sidebar.checkbox("Fit calibration offset", value=True)

st.markdown('<div class="evms-card">', unsafe_allow_html=True)
st.subheader("Data Inputs")
meas_file = st.file_uploader("Measurements CSV (`x,y,z,value`)", type="csv")
grid_file = st.file_uploader("Grid file (`.npy` mask or `.obj` mesh)", type=["npy", "obj"])
frac_file = st.file_uploader("Fractures JSON (optional)", type="json")
calib_file = st.file_uploader("Calibration CSV (`x,y,z,cps/s`) (optional)", type="csv")
st.markdown("</div>", unsafe_allow_html=True)

default_origin = (0.0, 0.0, 0.0)
default_spacing = 1.0
default_dims = (10, 10, 10)
if grid_file is not None and grid_file.name.lower().endswith(".obj"):
    try:
        default_origin, default_spacing, default_dims = _estimate_defaults_from_obj(grid_file)
    except Exception as exc:
        st.warning(f"Could not infer defaults from OBJ: {exc}")

st.markdown('<div class="evms-card">', unsafe_allow_html=True)
st.subheader("Grid Definition")
col_g0, col_g1, col_g2 = st.columns(3)
with col_g0:
    origin_x = st.number_input("Origin X", value=float(default_origin[0]))
with col_g1:
    origin_y = st.number_input("Origin Y", value=float(default_origin[1]))
with col_g2:
    origin_z = st.number_input("Origin Z", value=float(default_origin[2]))

col_s0, col_s1, col_s2, col_s3 = st.columns(4)
with col_s0:
    spacing_scalar = st.number_input("Isotropic spacing", value=float(default_spacing), min_value=1e-6)
with col_s1:
    dims_x = st.number_input("Dims X", value=int(default_dims[0]), min_value=1, step=1)
with col_s2:
    dims_y = st.number_input("Dims Y", value=int(default_dims[1]), min_value=1, step=1)
with col_s3:
    dims_z = st.number_input("Dims Z", value=int(default_dims[2]), min_value=1, step=1)
st.markdown("</div>", unsafe_allow_html=True)

st.expander("Model equations", expanded=False).markdown(
    "\n".join(
        [
            "Forward model:",
            r"$$M(\mathbf{x}) = \int_V S(\mathbf{r})\,G(\mathbf{x},\mathbf{r})\,\exp(-\mu\|\mathbf{x}-\mathbf{r}\|)\,dV + \varepsilon$$",
            r"$$G(\mathbf{x},\mathbf{r}) = \frac{1}{\|\mathbf{x}-\mathbf{r}\|^2 + \epsilon}$$",
            "Inversion objective:",
            r"$$\hat{S}=\arg\min_S \|AS-M\|_2^2 + \lambda\|LS\|_2^2$$",
        ]
    )
)

can_run = meas_file is not None and grid_file is not None
run_clicked = st.button("Run inversion", type="primary", disabled=not can_run)

if run_clicked:
    temp_paths = []
    try:
        measurement_csv = _uploaded_to_temp(meas_file, ".csv", text_mode=True)
        temp_paths.append(measurement_csv)

        if grid_file.name.lower().endswith(".npy"):
            grid_kind = "npy"
            grid_path = _uploaded_to_temp(grid_file, ".npy", text_mode=False)
            mesh_bytes = None
        else:
            grid_kind = "obj"
            grid_path = _uploaded_to_temp(grid_file, ".obj", text_mode=False)
            mesh_bytes = grid_file.getvalue()
        temp_paths.append(grid_path)

        fracture_path = None
        if frac_file is not None:
            fracture_path = _uploaded_to_temp(frac_file, ".json", text_mode=True)
            temp_paths.append(fracture_path)

        calibration_path = None
        if calib_file is not None:
            calibration_path = _uploaded_to_temp(calib_file, ".csv", text_mode=True)
            temp_paths.append(calibration_path)

        payload = {
            "measurement_csv": measurement_csv,
            "grid_path": grid_path,
            "grid_kind": grid_kind,
            "fracture_path": fracture_path,
            "calibration_path": calibration_path,
            "origin": (origin_x, origin_y, origin_z),
            "spacing": spacing_scalar,
            "dims": (int(dims_x), int(dims_y), int(dims_z)),
            "mu": mu,
            "r_max": r_max,
            "lam_manual": lam_manual,
            "use_auto_lam": use_auto_lam,
            "auto_tune_forward": auto_tune_forward,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "mu_steps": mu_steps,
            "rmax_min": rmax_min,
            "rmax_max": rmax_max,
            "rmax_steps": rmax_steps,
            "tuning_objective": "holdout" if tuning_objective == "Holdout RMSE" else "residual",
            "use_holdout_diagnostics": use_holdout_diagnostics,
            "holdout_fraction": holdout_fraction,
            "holdout_seed": int(holdout_seed),
            "use_calibration": use_calibration,
            "fit_calib_offset": fit_calib_offset,
            "mesh_bytes": mesh_bytes,
        }

        with st.spinner("Running EVMS inversion..."):
            st.session_state["evms_result"] = _run_inversion(payload)
    except Exception as exc:
        st.error(f"Inversion failed: {exc}")
    finally:
        for path in temp_paths:
            if path is not None and os.path.exists(path):
                os.unlink(path)

result = st.session_state.get("evms_result")
if result is None:
    st.info("Upload measurements and a grid, then click **Run inversion**.")
    st.stop()

st.markdown("---")
st.subheader("Run Summary")

col_m0, col_m1, col_m2, col_m3 = st.columns(4)
col_m0.metric("Active voxels", f"{result['grid'].n_voxels}")
col_m1.metric("Data misfit ||AS - M||", f"{result['data_misfit_norm']:.4f}")
col_m2.metric("Regularization ||LS||", f"{result['reg_norm']:.4f}")
col_m3.metric("Trust score", f"{result['trust_score']:.3f}")

st.write(
    f"Forward parameters: `mu={result['mu_used']:.4f} 1/m`, `R_max={result['rmax_used']:.3f} m`, "
    f"`lambda={result['lam']:.5g}`"
)
_render_mu_warnings(result, auto_tune_forward)

if result["calibration_model"] is not None:
    model = result["calibration_model"]
    st.success(
        "Calibration active: "
        f"gain={model.gain:.6g}, offset={model.offset:.6g}, R^2={model.r2:.4f}, n={model.n_samples}"
    )

if result["trust_level"] == "High":
    st.success(f"Trust level: {result['trust_level']} (score={result['trust_score']:.3f})")
elif result["trust_level"] == "Moderate":
    st.warning(f"Trust level: {result['trust_level']} (score={result['trust_score']:.3f})")
else:
    st.error(f"Trust level: {result['trust_level']} (score={result['trust_score']:.3f})")

st.write(f"Fit ratio ||AS - M|| / ||M||: {result['fit_ratio']:.4f}")
st.write(f"Holdout diagnostics: {result['holdout_summary']}")

if result["tuning_table"] is not None:
    st.subheader("Forward Tuning Table")
    st.dataframe(
        {
            "mu": result["tuning_table"][:, 0],
            "R_max": result["tuning_table"][:, 1],
            "lambda": result["tuning_table"][:, 2],
            "score": result["tuning_table"][:, 3],
        },
        use_container_width=True,
    )

vol_tab, diag_tab, export_tab = st.tabs(["Reconstruction", "Diagnostics", "Export"])

with vol_tab:
    st.subheader("Reconstructed Source Intensity")
    unit = result["value_unit"]
    s_vals = result["S_display"]
    st.write(
        f"Unit: `{unit}` | min={s_vals.min():.2f}, max={s_vals.max():.2f}, mean={s_vals.mean():.2f}"
    )

    full_s = _build_full_volume(result["grid"], s_vals)

    col_slice, col_hist = st.columns([1.3, 1.0])
    with col_slice:
        slice_k = st.slider("Slice index k", 0, result["grid"].dims[2] - 1, min(5, result["grid"].dims[2] - 1))
        fig_slice, ax_slice = plt.subplots(figsize=(7.2, 5.4))
        im = ax_slice.imshow(full_s[:, :, slice_k], origin="lower", cmap="RdYlGn_r")
        ax_slice.set_title(f"Source intensity slice (k={slice_k})")
        ax_slice.set_xlabel("X index")
        ax_slice.set_ylabel("Y index")
        cbar = fig_slice.colorbar(im, ax=ax_slice)
        cbar.set_label(f"Source intensity ({unit})")
        st.pyplot(fig_slice)

    with col_hist:
        fig_hist, ax_hist = plt.subplots(figsize=(6.2, 4.8))
        ax_hist.hist(s_vals, bins=30, alpha=0.85)
        ax_hist.set_title("Value distribution")
        ax_hist.set_xlabel(f"Source intensity ({unit})")
        ax_hist.set_ylabel("Voxel count")
        st.pyplot(fig_hist)

    st.subheader("3D voxel view")
    centers = result["grid"].voxel_centers()
    fig_3d = go.Figure(
        data=go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=s_vals,
                colorscale="RdYlGn",
                reversescale=True,
                colorbar=dict(title=f"Source intensity ({unit})"),
                showscale=True,
            ),
        )
    )
    fig_3d.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=35),
        title="Reconstructed source intensity field",
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with diag_tab:
    st.subheader("Residual Diagnostics")
    residuals = result["residuals"]
    points = result["points"]

    fig_residual_3d = go.Figure(
        data=go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(
                size=4,
                color=residuals,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Residual (M - AS)"),
                showscale=True,
            ),
        )
    )
    fig_residual_3d.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=35),
        title="Residual map at measurement points",
    )
    st.plotly_chart(fig_residual_3d, use_container_width=True)

    fig_res, ax_res = plt.subplots(figsize=(7.2, 4.8))
    ax_res.hist(residuals, bins=35, alpha=0.85)
    ax_res.set_title("Residual distribution")
    ax_res.set_xlabel("Residual (M - AS)")
    ax_res.set_ylabel("Count")
    st.pyplot(fig_res)

with export_tab:
    st.subheader("Export Outputs")

    col_export_0, col_export_1 = st.columns(2)
    with col_export_0:
        if st.button("Export source field (.npy)", use_container_width=True):
            save_grid(result["grid"], result["S_display"], "S_hat.npy")
            st.success("Saved `S_hat.npy` in the project root.")

    with col_export_1:
        if result["mesh_bytes"] is None:
            st.info("Mesh texture export is available only when the grid input is an OBJ file.")
        elif st.button("Export textured mesh (OBJ+MTL+PNG)", use_container_width=True):
            with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as handle:
                handle.write(result["mesh_bytes"])
                obj_temp = handle.name
            mesh = trimesh.load(obj_temp)
            os.unlink(obj_temp)
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    textured_mesh = apply_radioactivity_texture(mesh, result["grid"], result["S_display"], image_size=1024)
                    obj_path = os.path.join(tmpdir, "radioactivity_mesh.obj")
                    export_textured_obj(textured_mesh, obj_path, textured_mesh.visual.material.image)
                    st.success("Mesh export prepared.")
                    for ext in ("obj", "mtl", "png"):
                        for name in sorted(p for p in os.listdir(tmpdir) if p.lower().endswith(f".{ext}")):
                            path = os.path.join(tmpdir, name)
                            with open(path, "rb") as fobj:
                                st.download_button(
                                    f"Download {name}",
                                    fobj,
                                    file_name=name,
                                    use_container_width=True,
                                )
                except Exception as exc:
                    st.error(f"Texture export failed: {exc}")

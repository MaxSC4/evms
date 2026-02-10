"""How EVMS works page."""

import streamlit as st

from streamlit_ui import inject_global_css, render_page_header


inject_global_css()
render_page_header(
    "How EVMS Works",
    "Scientific model, assumptions, and inversion workflow",
)

st.markdown('<div class="evms-card">', unsafe_allow_html=True)
st.markdown(
    """
EVMS reconstructs a volumetric source field from surface gamma measurements under an attenuated kernel model.
The objective is not only data fit, but also geological coherence through graph-based regularization.
    """
)
st.markdown("</div>", unsafe_allow_html=True)

col_a, col_b = st.columns([1.2, 1.0])

with col_a:
    st.subheader("1) Forward model")
    st.latex(r"M(\mathbf{x}) = \int_V S(\mathbf{r}) G(\mathbf{x},\mathbf{r}) e^{-\mu\|\mathbf{x}-\mathbf{r}\|} dV + \varepsilon")
    st.latex(r"G(\mathbf{x},\mathbf{r}) = \frac{1}{\|\mathbf{x}-\mathbf{r}\|^2 + \epsilon}")
    st.markdown("Discrete voxel form:")
    st.latex(
        r"M \approx AS + \varepsilon,\qquad "
        r"A_{ij} = \frac{\Delta V_j}{|x_i-r_j|^2 + \epsilon}e^{-\mu|x_i-r_j|}"
    )
    st.latex(r"A_{ij}=0 \quad \text{when} \quad |x_i-r_j| > R_{\max}")

    st.subheader("2) Inverse problem")
    st.latex(r"\hat{S} = \arg\min_S \|AS - M\|_2^2 + \lambda\|LS\|_2^2")
    st.markdown("`L` is built from voxel-neighbor differences so that:")
    st.latex(r"\|LS\|_2^2 = \sum_{j\sim j'} w(j,j')(S_j-S_{j'})^2")

with col_b:
    st.subheader("3) Geological constraints")
    st.markdown(
        r"""
- **Layer-aware smoothing:** baseline edge weight is applied within the same stratigraphic layer.
- **Finite fracture barriers:** if a voxel edge crosses a fracture patch, the edge penalty increases.
- **Length/offset law:** fracture contribution follows:
        """
    )
    st.latex(r"\rho(s) = \rho_0 + \rho_1\frac{|s|}{L/2}, \qquad D(s)=\frac{L}{\max(\rho(s),\rho_{\min})}")
    st.markdown("with a monotonic coupling term:")
    st.latex(r"g(D)")
    st.markdown("typically:")
    st.latex(r"g(D)=D/D_{ref}")

    st.subheader("4) Diagnostics and trust")
    st.markdown(
        r"""
EVMS reports:
- data misfit \(\|AS-M\|\),
- regularization norm \(\|LS\|\),
- optional holdout RMSE,
- residual maps and trust level summary.
        """
    )

st.markdown("---")
st.subheader("Workflow Summary")
st.markdown(
    """
1. Upload measurements (`x,y,z,value`) and grid definition (`.npy` mask or `.obj` mesh).
2. Configure physics (`mu`, `R_max`) and regularization (`lambda`) manually or with auto-search.
3. Run inversion and inspect reconstructed slices, 3D field, and residual diagnostics.
4. Optionally apply calibration to convert relative source intensity into physical units.
5. Export volumetric field (`.npy`) and textured mesh (`OBJ + MTL + PNG`).
    """
)

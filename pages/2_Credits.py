"""Credits and acknowledgements page."""

import streamlit as st

from streamlit_ui import inject_global_css, render_page_header


inject_global_css()
render_page_header(
    "Credits",
    "Contributors, affiliations, and citation",
)

st.markdown('<div class="evms-card">', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Team")
    st.markdown(
        """
- **Main Developer:** Maxime SOARES CORREIA
- **Tutor / Scientific Supervisor:** Bertrand SAINT-BEZAR
        """
    )

    st.subheader("Institutional Affiliation")
    st.markdown(
        """
- **Laboratory:** GEOPS — Géosciences Paris-Saclay
- **University:** Université Paris-Saclay
- **Research Program:** URALOD
        """
    )

with col_b:
    st.subheader("Citation")
    st.markdown("Coming soon")

st.info(
    "For correspondence: soarescorreia@ipgp.fr and bertrand.saint-bezar@universite-paris-saclay.fr"
)

"""EVMS Streamlit entrypoint with official sidebar page navigation."""

import streamlit as st


st.set_page_config(
    page_title="EVMS",
    page_icon="🌍",
    layout="wide",
)

pages = [
    st.Page("pages/0_Inversion_Workspace.py", title="Inversion Workspace", icon="🌍", default=True),
    st.Page("pages/1_How_EVMS_Works.py", title="How EVMS Works", icon="📘"),
    st.Page("pages/2_Credits.py", title="Credits", icon="🧾"),
]

navigation = st.navigation(pages)
navigation.run()

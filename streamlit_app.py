"""EVMS Streamlit entrypoint with official sidebar page navigation."""

import hmac
import os
from typing import Optional

from streamlit_bootstrap import ensure_project_on_path

import streamlit as st


ensure_project_on_path()


def _configured_password() -> Optional[str]:
    try:
        secret = st.secrets.get("APP_PASSWORD")
    except Exception:
        secret = None
    if secret:
        return str(secret)

    env_value = os.getenv("EVMS_APP_PASSWORD")
    if env_value:
        return env_value
    return None


def _password_gate() -> None:
    expected_password = _configured_password()
    if not expected_password:
        return

    if st.session_state.get("evms_authenticated", False):
        return

    st.title("EVMS")
    st.caption("Protected access")
    st.info("This application is password-protected.")

    with st.form("evms_password_gate"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open EVMS")

    if submitted:
        if hmac.compare_digest(password, expected_password):
            st.session_state["evms_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


_password_gate()


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

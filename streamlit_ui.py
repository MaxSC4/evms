"""Shared Streamlit UI helpers for EVMS pages."""

import base64
from pathlib import Path
from typing import Optional

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = ROOT_DIR / "assets"
FONTS_DIR = ROOT_DIR / "fonts"
LOGO_CANDIDATES = [
    ASSETS_DIR / "geops_logo_nobg.png",
    ASSETS_DIR / "geops_lab_logo.png",
    ASSETS_DIR / "geops_lab_logo.jpg",
    ROOT_DIR / "geops_logo_nobg.png",
    ASSETS_DIR / "lab_logo_placeholder.svg",
]


def _encode_font(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    raw = path.read_bytes()
    return base64.b64encode(raw).decode("ascii")


def _font_face_css() -> str:
    regular_b64 = _encode_font(FONTS_DIR / "OpenSans-Regular.ttf")
    semibold_b64 = _encode_font(FONTS_DIR / "OpenSans-SemiBold.ttf")
    bold_b64 = _encode_font(FONTS_DIR / "OpenSans-Bold.ttf")
    if regular_b64 is None:
        return ""

    css = [
        "@font-face {",
        "  font-family: 'EVMSOpenSans';",
        "  font-style: normal;",
        "  font-weight: 400;",
        f"  src: url(data:font/ttf;base64,{regular_b64}) format('truetype');",
        "}",
    ]
    if semibold_b64 is not None:
        css.extend(
            [
                "@font-face {",
                "  font-family: 'EVMSOpenSans';",
                "  font-style: normal;",
                "  font-weight: 600;",
                f"  src: url(data:font/ttf;base64,{semibold_b64}) format('truetype');",
                "}",
            ]
        )
    if bold_b64 is not None:
        css.extend(
            [
                "@font-face {",
                "  font-family: 'EVMSOpenSans';",
                "  font-style: normal;",
                "  font-weight: 700;",
                f"  src: url(data:font/ttf;base64,{bold_b64}) format('truetype');",
                "}",
            ]
        )
    return "\n".join(css)


def inject_global_css() -> None:
    """Inject a cohesive publication-oriented style."""
    font_css = _font_face_css()
    css = """
<style>
__FONT_CSS__

:root {
  --evms-bg-0: #0f252e;
  --evms-bg-1: #0a1a22;
  --evms-ink: #e8f3f5;
  --evms-muted: #a7b8bd;
  --evms-accent: #7fb83d;
  --evms-accent-2: #8ac442;
  --evms-card: rgba(255, 255, 255, 0.10);
  --evms-border: rgba(138, 196, 66, 0.22);
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: "EVMSOpenSans", "Trebuchet MS", sans-serif;
  color: var(--evms-ink);
  background: linear-gradient(180deg, var(--evms-bg-0) 0%, var(--evms-bg-1) 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--evms-bg-0) 0%, var(--evms-bg-1) 100%);
}

[data-testid="stSidebar"] * {
  color: #eff7f8 !important;
}

[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div div {
  background: var(--evms-accent) !important;
}

[data-baseweb="checkbox"] [data-checked="true"],
[data-baseweb="radio"] [aria-checked="true"],
[data-baseweb="switch"] > label > div[data-checked="true"] {
  background-color: var(--evms-accent) !important;
  border-color: var(--evms-accent) !important;
}

[data-baseweb="checkbox"] input:checked + div,
[data-baseweb="switch"] input:checked + div {
  background-color: var(--evms-accent) !important;
  border-color: var(--evms-accent) !important;
}

[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
  background-color: var(--evms-accent) !important;
  border-color: var(--evms-accent) !important;
}

[data-testid="stPageLink-NavLink"] a,
[data-testid="stPageLink-NavLink"] span,
.evms-nav a {
  color: var(--evms-accent-2) !important;
  font-weight: 600;
}

.evms-hero {
  padding: 1.2rem 1.4rem;
  border: 1px solid var(--evms-border);
  border-radius: 18px;
  background: linear-gradient(140deg, rgba(255,255,255,0.10), rgba(255,255,255,0.06));
  box-shadow: 0 8px 24px rgba(2, 8, 10, 0.22);
}

.evms-hero h1 {
  margin: 0;
  font-size: 2.1rem;
  font-weight: 700;
  letter-spacing: -0.02em;
}

.evms-hero p {
  margin: 0.3rem 0 0;
  color: var(--evms-muted);
  font-size: 0.98rem;
}

.evms-card {
  border: none;
  border-top: 1px solid rgba(138, 196, 66, 0.32);
  border-radius: 0;
  padding: 0.70rem 0 0.10rem 0;
  background: transparent;
  box-shadow: none;
}

.evms-logo-placeholder {
  border: 1px dashed rgba(138, 196, 66, 0.6);
  border-radius: 14px;
  padding: 0.85rem;
  text-align: center;
  color: #d3e5e9;
  font-size: 0.92rem;
  background: rgba(255,255,255,0.06);
}

code {
  font-family: "IBM Plex Mono", "Menlo", monospace;
}

[data-testid="stMetricValue"] {
  font-weight: 700;
}

hr {
  border: none;
  border-top: 1px solid rgba(138, 196, 66, 0.25);
  margin-top: 0.9rem;
  margin-bottom: 0.9rem;
}
</style>
    """
    st.markdown(css.replace("__FONT_CSS__", font_css), unsafe_allow_html=True)


def _resolve_logo_path() -> Optional[Path]:
    for candidate in LOGO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def render_logo_placeholder() -> None:
    """Render laboratory logo if found, else show clear placeholder."""
    logo_path = _resolve_logo_path()
    if logo_path is not None:
        st.image(str(logo_path), use_container_width=True)
        return

    st.markdown(
        """
<div class="evms-logo-placeholder">
  <strong>Laboratory logo placeholder</strong><br/>
  Add the logo file at:<br/><code>assets/geops_logo_nobg.png</code>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render branded page header."""
    col_logo, col_title = st.columns([1, 3], vertical_alignment="center")
    with col_logo:
        render_logo_placeholder()
    with col_title:
        st.markdown(
            f"""
<div class="evms-hero">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
            """,
            unsafe_allow_html=True,
        )

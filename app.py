"""
방송 심의 AI Agent — MVP Entry Point

Run:
    streamlit run app.py
"""

import sys

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="방송 심의 AI Agent",
    page_icon="🎬",
    layout="wide",
)

# ── Global CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ========================================
   Soft Indigo Pastel Theme
   ======================================== */

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: #F8F7FF;
}

/* ── Top header bar ── */
header[data-testid="stHeader"] {
    background: rgba(248, 247, 255, 0.85) !important;
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(123, 111, 232, 0.15);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F0EEFF 0%, #E8E3FF 100%) !important;
    border-right: 1px solid rgba(123, 111, 232, 0.2);
}
section[data-testid="stSidebar"] > div {
    padding-top: 0.75rem;
}

/* ── Headings ── */
h1 {
    color: #1E1B4B !important;
    font-weight: 800 !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.5px;
    padding-bottom: 0.5rem !important;
    border-bottom: 2px solid #C7C2F8;
}
h2 {
    color: #2E2B5A !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}
h3 {
    color: #3D3875 !important;
    font-weight: 600 !important;
}

/* ── Primary Buttons ── */
button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7B6FE8 0%, #5B4ED0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.2px;
    box-shadow: 0 2px 8px rgba(91, 78, 208, 0.3) !important;
    transition: all 0.2s ease !important;
}
button[data-testid="baseButton-primary"]:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #6A5FD8 0%, #4A3DC0 100%) !important;
    box-shadow: 0 4px 16px rgba(91, 78, 208, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary Buttons ── */
button[data-testid="baseButton-secondary"],
.stButton > button[kind="secondary"] {
    background: #fff !important;
    border: 1.5px solid #C7C2F8 !important;
    border-radius: 10px !important;
    color: #5B4ED0 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
}
button[data-testid="baseButton-secondary"]:hover,
.stButton > button[kind="secondary"]:hover {
    background: #F0EEFF !important;
    border-color: #7B6FE8 !important;
}

/* ── Form Submit Button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #7B6FE8 0%, #5B4ED0 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2.2rem !important;
    box-shadow: 0 2px 8px rgba(91, 78, 208, 0.3) !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(91, 78, 208, 0.4) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    background: #fff !important;
    border: 1px solid #DDD8FF !important;
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(123, 111, 232, 0.08);
}
div[data-testid="stExpander"] > details > summary {
    background: #F5F3FF !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    color: #3D3875 !important;
    padding: 0.75rem 1rem !important;
}
div[data-testid="stExpander"] > details > summary:hover {
    background: #EDE9FF !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #fff !important;
    border: 1px solid #DDD8FF;
    border-left: 4px solid #7B6FE8 !important;
    border-radius: 12px;
    padding: 14px 18px !important;
    box-shadow: 0 2px 8px rgba(123, 111, 232, 0.1);
}
div[data-testid="stMetricLabel"] {
    color: #6B64A8 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetricValue"] {
    color: #1E1B4B !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}

/* ── Text inputs & Textarea ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    border: 1.5px solid #DDD8FF !important;
    border-radius: 10px !important;
    background: #fff !important;
    color: #2E2B5A !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7B6FE8 !important;
    box-shadow: 0 0 0 3px rgba(123, 111, 232, 0.15) !important;
}

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border: 1.5px solid #DDD8FF !important;
    border-radius: 10px !important;
    background: #fff !important;
}

/* ── File Uploader ── */
div[data-testid="stFileUploader"] {
    border: 2px dashed #C7C2F8 !important;
    border-radius: 14px !important;
    background: #F5F3FF !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #7B6FE8 !important;
}

/* ── Alert boxes ── */
div[data-baseweb="notification"] {
    border-radius: 10px !important;
}

/* ── Divider ── */
hr {
    border-color: #E0DCFF !important;
    opacity: 0.8;
}

/* ── Code blocks ── */
code {
    background: #EDE9FF !important;
    border-radius: 5px !important;
    color: #5B4ED0 !important;
    padding: 2px 7px !important;
    font-size: 0.82em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F0EEFF; }
::-webkit-scrollbar-thumb { background: #C7C2F8; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7B6FE8; }

/* ── Spinner ── */
div[data-testid="stSpinner"] > div {
    border-top-color: #7B6FE8 !important;
}

/* ── Caption text ── */
.stCaption, small {
    color: #8B86C0 !important;
}

/* ── DataFrame / table ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #DDD8FF !important;
}
</style>
""", unsafe_allow_html=True)


# ── Database initialisation ──
from storage.database import init_db  # noqa: E402

init_db()
print("[DEBUG] init_db 완료", flush=True)

# ── Session state defaults ──
if "current_page" not in st.session_state:
    st.session_state.current_page = "knowledge"

# ── Page registry ──
MAIN_PAGES: list[tuple[str, str]] = [
    ("📚 기준지식 관리", "knowledge"),
    ("📝 심의요청 등록", "request"),
    ("📋 심의요청 목록", "list"),
]

# ── Sidebar navigation ──
with st.sidebar:
    # 브랜드 배너
    st.markdown("""
    <div style="
        margin: 0 0.5rem 0.5rem 0.5rem;
        padding: 18px 16px 16px;
        background: linear-gradient(135deg, #fff 0%, #F5F3FF 100%);
        border: 1px solid #DDD8FF;
        border-radius: 16px;
        box-shadow: 0 2px 12px rgba(123, 111, 232, 0.12);
        text-align: center;
    ">
        <div style="
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 48px; height: 48px;
            background: linear-gradient(135deg, #7B6FE8, #5B4ED0);
            border-radius: 14px;
            font-size: 1.4rem;
            box-shadow: 0 4px 12px rgba(91, 78, 208, 0.35);
            margin-bottom: 10px;
        ">🎬</div>
        <div style="
            font-size: 0.95rem;
            font-weight: 800;
            color: #1E1B4B;
            line-height: 1.3;
            letter-spacing: -0.3px;
        ">방송 심의 AI Agent</div>
        <div style="
            margin-top: 6px;
            display: inline-block;
            font-size: 0.65rem;
            font-weight: 600;
            color: #7B6FE8;
            background: #EDE9FF;
            padding: 2px 10px;
            border-radius: 20px;
            letter-spacing: 0.5px;
        ">MVP v0.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    for label, page_key in MAIN_PAGES:
        is_current = st.session_state.current_page == page_key
        if st.button(
            label,
            key=f"nav_{page_key}",
            use_container_width=True,
            type="primary" if is_current else "secondary",
        ):
            if not is_current:
                st.session_state.current_page = page_key
                st.rerun()

    # Detail page back button
    if st.session_state.current_page == "detail":
        st.divider()
        st.caption("🔍 심의 상세 화면")
        if st.button(
            "← 목록으로 돌아가기",
            key="nav_back",
            use_container_width=True,
        ):
            st.session_state.current_page = "list"
            st.rerun()

    # 사이드바 하단
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 1.5rem;
        left: 0;
        width: 240px;
        text-align: center;
        font-size: 0.62rem;
        color: #A09CC8;
        padding: 0 1rem;
        line-height: 1.6;
    ">
        ⚡ Powered by LangGraph &amp; Claude<br>
        <span style="opacity: 0.6;">방송통신심의위원회 규정 기반</span>
    </div>
    """, unsafe_allow_html=True)


# ── Import renderers ──
from ui.page_knowledge import render as render_knowledge  # noqa: E402
from ui.page_request import render as render_request      # noqa: E402
from ui.page_list import render as render_list            # noqa: E402
from ui.page_review_detail import render as render_detail # noqa: E402

RENDERERS: dict[str, callable] = {
    "knowledge": render_knowledge,
    "request":   render_request,
    "list":      render_list,
    "detail":    render_detail,
}

# ── Render current page ──
RENDERERS[st.session_state.current_page]()

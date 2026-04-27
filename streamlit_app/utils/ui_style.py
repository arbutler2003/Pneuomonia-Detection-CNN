import streamlit as st


def apply_theme() -> None:
    """Inject a lightweight, consistent visual theme for the app."""
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 10% 0%, #162033 0%, #0b1220 45%, #070d18 100%);
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stSidebar"] {
                background: #0a1222;
                border-right: 1px solid rgba(148, 163, 184, 0.2);
            }

            [data-testid="stSidebar"] * {
                color: #e2e8f0;
            }

            [data-testid="stSidebar"] .stFileUploader label,
            [data-testid="stSidebar"] .stSelectbox label {
                color: #e2e8f0;
                font-weight: 500;
                font-size: 0.84rem;
            }

            [data-testid="stSidebarNav"] {
                margin-top: 0.3rem;
            }

            [data-testid="stSidebarNav"] a {
                border-radius: 8px;
                margin-bottom: 0.15rem;
                font-size: 1rem;
                font-weight: 600;
                padding-top: 0.35rem;
                padding-bottom: 0.35rem;
            }

            [data-testid="stSidebarNav"] a:hover {
                background-color: rgba(148, 163, 184, 0.12);
            }

            .sidebar-section-title {
                margin: 0.25rem 0 0.5rem 0;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #94a3b8;
                font-weight: 700;
            }

            .sidebar-meta {
                margin: 0 0 0.45rem 0;
                display: flex;
                justify-content: space-between;
                gap: 0.6rem;
                font-size: 0.82rem;
                color: #cbd5e1;
            }

            .sidebar-meta span {
                color: #94a3b8;
            }

            .sidebar-meta strong {
                color: #e2e8f0;
                font-weight: 600;
            }

            .sidebar-muted {
                margin: 0.15rem 0 0.55rem 0;
                font-size: 0.8rem;
                color: #94a3b8;
                line-height: 1.35;
            }

            .sidebar-emphasis {
                color: #cbd5e1;
            }

            .block-container {
                max-width: 1200px;
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }

            h1, h2, h3 {
                letter-spacing: -0.01em;
                color: #f8fafc;
            }

            p, li {
                color: #cbd5e1;
                line-height: 1.6;
            }

            [data-testid="stMarkdownContainer"] {
                color: #cbd5e1;
            }

            [data-testid="stMarkdownContainer"] code {
                border-radius: 8px;
                background: rgba(30, 41, 59, 0.75);
                color: #e2e8f0;
            }

            [data-testid="stDataFrame"] {
                border-radius: 12px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.28);
                box-shadow: 0 10px 24px rgba(2, 6, 23, 0.45);
            }

            [data-testid="stImage"] img {
                border-radius: 12px;
            }

            [data-testid="stVerticalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(100, 116, 139, 0.45);
                border-radius: 14px;
                box-shadow: 0 10px 26px rgba(2, 6, 23, 0.38);
            }

            .stAlert {
                border-radius: 12px;
                border: 1px solid rgba(100, 116, 139, 0.45);
                background: rgba(15, 23, 42, 0.88);
            }

            .stButton button {
                border-radius: 10px;
                border: 1px solid #60a5fa;
                background: linear-gradient(135deg, #2563eb, #1e3a8a);
                color: white;
                font-weight: 600;
            }

            .stButton button:hover {
                border-color: #93c5fd;
                background: linear-gradient(135deg, #1d4ed8, #1e40af);
            }

            .live-demo-header h1 {
                margin: 0;
                font-size: 2rem;
                line-height: 1.2;
                color: #f8fafc;
            }

            .live-demo-header p {
                margin: 0.35rem 0 0 0;
                font-size: 1.28rem;
                color: #dbeafe;
                font-weight: 600;
            }

            .live-demo-header span {
                display: block;
                margin-top: 0.45rem;
                color: #94a3b8;
                font-size: 0.92rem;
            }

            .live-demo-status {
                margin: 0.25rem 0 1rem 0;
                padding: 0.55rem 0.8rem;
                border: 1px solid rgba(96, 165, 250, 0.35);
                background: rgba(30, 64, 175, 0.2);
                border-radius: 10px;
                color: #dbeafe;
                font-size: 0.92rem;
            }

            .diagnosis-card {
                padding: 1rem 1.1rem;
                border-radius: 12px;
                background: rgba(15, 23, 42, 0.9);
                border: 1px solid rgba(100, 116, 139, 0.35);
                border-left: 6px solid;
                margin-bottom: 0.9rem;
                min-height: 132px;
            }

            .diagnosis-card .diagnosis-label {
                margin: 0;
                color: #93c5fd;
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }

            .diagnosis-card h2 {
                margin: 0.45rem 0 0.3rem 0;
                font-size: 2rem;
                line-height: 1.1;
            }

            .diagnosis-card .diagnosis-confidence {
                margin: 0;
                color: #e2e8f0;
                font-size: 1rem;
                font-weight: 500;
            }

            .diagnosis-card-muted {
                border-left-color: rgba(148, 163, 184, 0.5);
            }

            .diagnosis-card-muted h2 {
                color: #cbd5e1;
            }

            .source-hint-card {
                margin-top: 0.55rem;
                padding: 0.7rem 0.85rem;
                border: 1px solid rgba(100, 116, 139, 0.45);
                background: rgba(15, 23, 42, 0.88);
                border-radius: 10px;
                color: #dbeafe;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

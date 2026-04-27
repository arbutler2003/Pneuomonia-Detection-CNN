import streamlit as st


def render_sidebar() -> None:
    """Render a minimal, consistent sidebar across pages."""
    with st.sidebar:
        st.markdown('<p class="sidebar-section-title">Application Details</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="sidebar-meta"><span>Developer</span><strong>Aiden Butler</strong></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sidebar-meta"><span>Version</span><strong>1.0.0</strong></p>',
            unsafe_allow_html=True,
        )

import streamlit as st


def render_lime(lime_html: str):
    """
    Render hasil LIME ke Streamlit
    """

    st.markdown("### 🔍 LIME Explanation")

    st.components.v1.html(
        lime_html,
        height=400,
        scrolling=True
    )
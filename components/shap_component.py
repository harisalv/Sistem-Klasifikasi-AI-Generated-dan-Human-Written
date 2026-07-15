import streamlit as st

def render_shap(shap_html: str):
    st.markdown("### 🧠 SHAP Explanation")

    st.components.v1.html(
        shap_html,
        height=450,
        scrolling=True
    )
import os

import streamlit as st
from utils.ui_style import apply_theme
from utils.sidebar import render_sidebar

# --- Page Configuration ---
st.set_page_config(
    page_title="Pneumonia Detection | Capstone",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

render_sidebar()

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")

# --- Main Landing Page ---
st.title("Automated Pneumonia Detection & Architectural Benchmark")
st.markdown("### End-to-End Machine Learning Pipeline & Evaluation")
st.write(
    "This application presents an end-to-end machine learning workflow for "
    "pediatric pneumonia screening. The pipeline prepares the dataset and "
    "utilizes transfer learning to build a custom CNN model, it is then "
    "benchmarked against enterprise AutoML and multimodal LLM approaches to "
    "evaluate trade-offs between raw accuracy, clinical interpretability, and "
    "structured output generation."
)
st.divider()

# --- The Three Pillars Section ---
st.header("A Three-Pillar Comparative Analysis")
st.write("")

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Custom CNN (Fine-Tuned)")
        st.write(
            "A transfer-learning architecture built on a fine-tuned "
            "**ResNet50** model with a custom head in Keras. It emphasizes "
            "explainability with Grad-CAM and allows for transparency and "
            "modification of the model architecture."
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Enterprise Cloud (Vertex AI)")
        st.write(
            "A commercial benchmark using Google's proprietary **Neural "
            "Architecture Search (NAS)**. This represents a black-box approach "
            "optimized for pure metrics on the sanitized dataset."
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 3. Multimodal LLM (Gemini API)")
        st.write(
            "A generative AI baseline using zero-shot Visual Chain-of-Thought "
            "(vCoT) prompting. It evaluates how well LLM models can produce "
            "structured clinical-style JSON reasoning from the sanitized "
            "dataset."
        )

st.divider()

# --- Methodology & The Clever Hans Problem ---
st.header("Methodology: Mitigating Artifact Bias")

problem_col, solution_col = st.columns([1, 1], gap="large")

with problem_col:
    with st.container(border=True):
        st.markdown("##### The Problem")
        st.markdown(
            "In this dataset, there are two cases for how the X-rays are taken: **standing** (arms up, 'R' indicator below right arm) and **bedridden** (arms down, 'R' indicator above right arm). Bedridden patients are far more likely to have pneumonia, so early iterations of the model learned these as relevant features."
        )

with solution_col:
    with st.container(border=True):
        st.markdown("##### The Solution")
        st.markdown(
            "**Initial Approach:** Tried using a U-net model to isolate the lungs in each image. The model failed to accurately isolate the lungs, as U-net models were trained on healthy adult lungs."
        )
        st.markdown(
            "**Final Approach:** Applied a 12% conservative crop to remove peripheral artifacts and isolate the thoracic cavity. This approach worked very well for this dataset, and introduces less chance for error, as the worst case is the outer edges are slightly cut off."
        )

with st.container(border=True):
    clever_hans_image_path = os.path.join(DATA_DIR, "clever_hans_example.png")
    if os.path.exists(clever_hans_image_path):
        st.image(
            clever_hans_image_path,
            caption="Sample Pneumonia Presentation",
            width="stretch",
        )
    else:
        st.warning(
            "Sample image not found at streamlit_app/data/clever_hans_example.png."
        )

st.write("")
st.success(
    "Next step: use the sidebar to open the Live Demo or Metrics Comparison page."
)
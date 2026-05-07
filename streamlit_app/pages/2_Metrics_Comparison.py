import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from utils.ui_style import apply_theme
from utils.sidebar import render_sidebar

# --- Dynamic Path Resolution ---
# This ensures images load regardless of where the app is hosted
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")

MATRIX_VALUES = {
    "custom_cnn": np.array([[174, 60], [5, 385]]),
    "vertex_ai": np.array([[229, 10], [4, 386]]),
    "gemini_api": np.array([[183, 46], [129, 259]]),
}


def render_confusion_matrix(matrix: np.ndarray, cmap: str) -> None:
    """Render a consistent confusion matrix with count + percentage labels."""
    total = matrix.sum()
    percentages = (matrix / total) * 100
    cell_labels = np.array(
        [
            ["True Negative", "False Positive"],
            ["False Negative", "True Positive"],
        ]
    )
    annotations = np.array(
        [
            [f"{cell_labels[r, c]}\n{matrix[r, c]}\n{percentages[r, c]:.1f}%" for c in range(matrix.shape[1])]
            for r in range(matrix.shape[0])
        ]
    )

    fig, ax = plt.subplots(figsize=(4.0, 4.0), dpi=150)
    sns.heatmap(
        matrix,
        annot=annotations,
        fmt="",
        cmap=cmap,
        cbar=False,
        square=True,
        linewidths=1.0,
        linecolor="#e5e7eb",
        xticklabels=["Normal", "Pneumonia"],
        yticklabels=["Normal", "Pneumonia"],
        ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("Actual Label", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

st.set_page_config(page_title="Architectural Benchmark", layout="wide")
apply_theme()
render_sidebar()

st.title("Architectural Benchmark: Performance vs. Interpretability")
st.markdown("### Quantitative and Qualitative Performance Analysis")
st.write(
    "A rigorous evaluation of three distinct AI architectures to demonstrate "
    "the business and clinical trade-offs between raw quantitative accuracy, "
    "model transparency, and unstructured-to-structured data processing."
)
st.divider()

# ------ PART 1: QUANTITATIVE -------
st.header("Quantitative Performance")
st.write("Comparing the raw classification power and error distribution of each architecture.")
st.write("")

metrics_data = {
    "Metric": ["Accuracy", "Recall (Sensitivity)", "Precision", "AUC"],
    "Custom CNN (Fine-Tuned)": ["91.99%", "98.46%", "89.72%", "0.9634"],
    "Vertex AI (AutoML)": ["97.77%", "98.97%", "97.47%", "0.9880"],
    "Gemini API (2.5 Flash)": ["71.64%", "84.92%", "66.75%", "0.7535"],
}

df = pd.DataFrame(metrics_data)
st.dataframe(df.set_index("Metric"), width="stretch")

st.write("")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown("#### 1. Custom CNN")
        st.caption("ResNet50 Base + Custom Head")
        render_confusion_matrix(MATRIX_VALUES["custom_cnn"], "Blues")
        st.caption("Custom CNN Matrix")
        st.markdown(
            "**Key Finding (High Sensitivity):** Achieved an exceptional Recall "
            "of 98.46%. In a clinical screening setting, optimizing for high "
            "sensitivity establishes a safety net by minimizing False Negatives."
        )

with col2:
    with st.container(border=True):
        st.markdown("#### 2. Vertex AI")
        st.caption("Google Proprietary NAS")
        render_confusion_matrix(MATRIX_VALUES["vertex_ai"], "Blues")
        st.caption("Vertex AI Matrix")
        st.markdown(
            "**Key Finding (Optimization):** "
            "Achieving a near-perfect 98.97% Recall alongside a 97.47% Precision, "
            "it aggressively minimizes False Negatives while representing the gold "
            "standard for out-of-the-box accuracy."
        )

with col3:
    with st.container(border=True):
        st.markdown("#### 3. Gemini API")
        st.caption("Gemini 2.5 Flash")
        render_confusion_matrix(MATRIX_VALUES["gemini_api"], "Blues")
        st.caption("Gemini 2.5 Flash Matrix")
        st.markdown(
            "**Key Finding (Zero-Shot Constraints):** Out-of-the-box multimodal "
            "models currently trail heavily in binary diagnostic accuracy "
            "(71.64%) and precision (66.75%)."
        )

st.divider()

# ------- PART 2: QUALITATIVE ---
st.header("Qualitative Analysis")
st.write(
    "Evaluating the clinical viability of each model beyond raw performance "
    "metrics (Transparency vs. Accuracy)."
)
st.write("")

top_col1, top_col2 = st.columns(2, gap="medium")

with top_col1:
    with st.container(border=True):
        st.markdown("#### The Interpretability Advantage")
        st.write(
            "Unlike commercial APIs, the Custom CNN offers full transparency. By "
            "accepting a slight reduction in raw accuracy compared to enterprise "
            "AutoML, the system gains the ability to explicitly highlight the "
            "regions driving the prediction. This provides clinicians with the "
            "necessary visual verification to safely accept the model's output."
        )

with top_col2:
    with st.container(border=True):
        st.markdown("#### The 'Black Box' Trade-off")
        st.write(
            "Google's Neural Architecture Search (NAS) provides elite, "
            "production-grade metrics. However, the architecture abstracts away "
            "the network layers, offering zero interpretability. Users receive a "
            "highly accurate probability score, but no mapping to validate why "
            "the diagnosis was made."
        )

st.write("")
with st.container(border=True):
    st.markdown("#### The Structured Data Advantage")
    st.write(
        "While the generative model failed as a binary classifier, it "
        "demonstrated massive potential for automated ETL pipelines. Instead of "
        "a probability score, the API synthesizes visual features into "
        "actionable clinical text. It successfully generates structured, "
        "descriptive reasoning ready for Electronic Medical Record (EMR) "
        "ingestion."
    )
    jsonl_path = os.path.join(DATA_DIR, "evaluation_results_gemini_evaluation_results.jsonl")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_record = json.loads(f.readline())
    st.code(json.dumps(first_record, indent=2), language="json")

st.divider()

st.header("Executive Summary")
st.info("""
This benchmark reveals distinct operational use-cases for each architecture. While **Vertex AI** delivers the most rigorous quantitative metrics, the **Custom CNN** remains the most viable clinical tool by mitigating the black-box problem via Grad-CAM transparency.

Furthermore, the **Gemini API** highlights the emerging value of multimodal models not as classifiers, but as powerful engines for structuring complex visual data into analyzable text formats.
""")
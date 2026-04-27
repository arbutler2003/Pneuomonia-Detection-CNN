import io
import os

import streamlit as st
import tensorflow as tf
from utils.image_processing import load_and_preprocess_image
from utils.gradcam import make_gradcam_heatmap, superimpose_heatmap
from utils.sidebar import render_sidebar
from utils.ui_style import apply_theme

# --- Page Config ---
st.set_page_config(page_title="Live Inference", page_icon="🔬", layout="wide")
apply_theme()


@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the trained Keras model used for inference."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "best_cropped_finetuned.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "Model file not found at streamlit_app/models/best_cropped_finetuned.keras."
        )
    return tf.keras.models.load_model(model_path)


try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

render_sidebar()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_pneumonia_path = os.path.join(base_dir, "data", "live_test_pneumonia.jpeg")
default_normal_path = os.path.join(base_dir, "data", "live_test_normal.jpeg")

# --- Main UI ---
st.title("Live Inference Dashboard")
st.markdown("### Real-Time Model Serving")
st.caption(
    "Evaluate the fine-tuned ResNet50 model in a production-style inference "
    "flow. The interface processes user input, applies automated preprocessing, "
    "runs inference, and returns a Grad-CAM map for transparent review."
)
st.divider()

card_col1, card_col2 = st.columns(2, gap="large")

with card_col1:
    st.subheader("Image Source")
    source_mode = st.selectbox(
        "Source",
        ["Default Pneumonia", "Default Normal", "Custom Upload"],
        index=0,
    )
    uploaded_file = None
    source_hint = ""
    if source_mode == "Custom Upload":
        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray (.jpg, .png)",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is None:
            source_hint = "Upload a chest X-ray image to run inference."
        else:
            source_hint = "Running custom uploaded image."
    elif source_mode == "Default Normal":
        source_hint = "Running default sample: live_test_normal.jpeg"
    else:
        source_hint = "Running default sample: live_test_pneumonia.jpeg"

    st.markdown(
        f"""
        <div class="source-hint-card">
            {source_hint}
        </div>
        """,
        unsafe_allow_html=True,
    )

input_source = None
if source_mode == "Default Pneumonia":
    if os.path.exists(default_pneumonia_path):
        with open(default_pneumonia_path, "rb") as sample_file:
            input_source = io.BytesIO(sample_file.read())
        input_source.name = "live_test_pneumonia.jpeg"
    else:
        st.error(
            "Default sample not found at streamlit_app/data/live_test_pneumonia.jpeg"
        )
elif source_mode == "Default Normal":
    if os.path.exists(default_normal_path):
        with open(default_normal_path, "rb") as sample_file:
            input_source = io.BytesIO(sample_file.read())
        input_source.name = "live_test_normal.jpeg"
    else:
        st.error("Default sample not found at streamlit_app/data/live_test_normal.jpeg")
else:
    input_source = uploaded_file

label = None
confidence = None
overlay_img = None
rgb_img = None

if input_source is not None:
    rgb_img, preprocessed_img = load_and_preprocess_image(input_source)

    if preprocessed_img is not None:
        with st.spinner("Running inference pipeline..."):
            prediction = model.predict(preprocessed_img, verbose=0)[0][0]
            label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            confidence = prediction if prediction > 0.5 else (1 - prediction)

        resnet_base = model.get_layer("resnet50")
        if "conv5_block3_out" not in [layer.name for layer in resnet_base.layers]:
            st.error(
                "Expected Grad-CAM layer 'conv5_block3_out' is missing in the model."
            )
            st.stop()

        heatmap = make_gradcam_heatmap(
            preprocessed_img,
            model,
            resnet_base,
            "conv5_block3_out",
        )
        overlay_img = superimpose_heatmap(rgb_img, heatmap)

with card_col2:
    st.subheader("Diagnostic Assessment")
    if label is not None and confidence is not None:
        card_color = "#E74C3C" if label == "PNEUMONIA" else "#2ECC71"
        st.markdown(
            f"""
            <div class="diagnosis-card" style="border-left-color: {card_color};">
                <p class="diagnosis-label">Diagnostic Assessment</p>
                <h2 style="color: {card_color};">{label}</h2>
                <p class="diagnosis-confidence">Model Confidence: {confidence:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="diagnosis-card diagnosis-card-muted">
                <p class="diagnosis-label">Diagnostic Assessment</p>
                <h2>Awaiting Input</h2>
                <p class="diagnosis-confidence">Select a source to run inference.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

if rgb_img is not None and overlay_img is not None and label is not None:
    st.subheader("Explainable AI: Feature Activation Mapping")
    st.caption(
        "In high-stakes clinical workflows, diagnostic models should not operate "
        "as black boxes. The Grad-CAM overlay highlights the regions that most "
        "influenced the final prediction, supporting clinician-facing verification."
    )
    xray_col1, xray_col2 = st.columns(2, gap="large")

    with xray_col1:
        with st.container(border=True):
            st.image(rgb_img, caption="Processed Input (12% Edge Crop)", width="stretch")

    with xray_col2:
        with st.container(border=True):
            st.image(overlay_img, caption=f"Grad-CAM Activation ({label})", width="stretch")
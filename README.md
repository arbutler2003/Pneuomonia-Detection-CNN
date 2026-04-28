# Automated Pneumonia Detection and Architectural Benchmark
**This application is live on Streamlit's community cloud at https://pneumonia-detection-cnn-capstone.streamlit.app/**

This project presents an end-to-end workflow for pediatric pneumonia screening and compares three AI approaches:
- Custom fine-tuned CNN (ResNet50 backbone)
- Vertex AI AutoML benchmark
- Gemini multimodal API baseline

The Streamlit app demonstrates live inference, Grad-CAM visualization, and quantitative/qualitative architecture comparisons.

## Repository Structure

- `streamlit_app/Home.py` - landing page and methodology context
- `streamlit_app/pages/1_Live_Demo.py` - live inference and Grad-CAM page
- `streamlit_app/pages/2_Metrics_Comparison.py` - benchmark metrics and qualitative analysis
- `streamlit_app/utils/` - shared preprocessing, Grad-CAM, sidebar, and styling helpers
- `chest_xray_pneumonia_detection.ipynb` - model development and evaluation notebook

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r streamlit_app/requirements.txt
```

3. Run the app:

```bash
streamlit run streamlit_app/Home.py
```

## Required Runtime Assets

The app expects these files at runtime:
- `streamlit_app/models/best_cropped_finetuned.keras (https://github.com/arbutler2003/Pneuomonia-Detection-CNN/releases/tag/v1.0.0)`
- `streamlit_app/data/live_test_pneumonia.png` (optional but used as default sample)
- `streamlit_app/data/clever_hans_example.png` (optional display image on home page)
- `streamlit_app/data/evaluation_results_gemini_evaluation_results.jsonl`

If optional image files are missing, the app shows a warning and continues running.

## Reproducibility Notes

- The notebook currently contains Colab-specific steps and paths.
- For local reproducibility, adapt dataset/model paths to your environment before running notebook cells.
- This repository includes the Streamlit application code and expects model/data artifacts to be provisioned locally.

## Known Constraints

- Grad-CAM visualization assumes a ResNet50-based model with `conv5_block3_out`.
- If model architecture changes, update the Grad-CAM target layer in `streamlit_app/pages/1_Live_Demo.py`.

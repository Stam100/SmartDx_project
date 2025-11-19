# streamlit_app.py
import streamlit as st
from utils import load_image_pil, preprocess_pil_image, load_model, predict, make_gradcam_heatmap, overlay_heatmap_on_image
import os
from datetime import datetime
from PIL import Image
import io

# ------------------------
# Configuration
# ------------------------
MODEL_PATH = os.path.join("models", "malaria_model.keras") 
st.set_page_config(page_title="SmartDx - Malaria Detector", layout="centered")

# ------------------------
# Load model (cached)
# ------------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}. Place your trained model there.")
        return None
    model = load_model(MODEL_PATH)
    return model

model = get_model()

# ------------------------
# UI Layout
# ------------------------
st.title("SmartDx — AI Malaria Detector")
st.markdown("Upload a microscope image (blood smear). The model will predict if the image is **Parasitized** or **Uninfected**.")
st.sidebar.header("About")
st.sidebar.info("This tool is a decision-support prototype. Do **not** use it as a sole clinical diagnostic. See PRD for details.")

# File uploader
uploaded_file = st.file_uploader("Upload image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Minimal logging (in-memory for session)
if "log" not in st.session_state:
    st.session_state["log"] = []

# Example images button (if one adds sample images to assets later)
# if st.button("Use sample image"):
#     uploaded_file = open("assets/sample.jpg", "rb")

if uploaded_file is not None and model is not None:
    # Load & show original image
    pil_img = load_image_pil(uploaded_file)
    st.image(pil_img, caption="Uploaded image", use_column_width=True)

    # Preprocess
    input_arr = preprocess_pil_image(pil_img)

    # Run prediction
    with st.spinner("Running model..."):
        label, confidence, prob = predict(model, input_arr)

    # Show result
    st.subheader("Result")
    st.metric(label=f"Prediction: {label}", value=f"{confidence*100:.1f}% confidence", delta=None)
    st.write(f"Raw probability (model sigmoid output): {prob:.4f}")

    # Show Grad-CAM heatmap (optional)
    if st.checkbox("Show explanation (Grad-CAM)"):
        heatmap = make_gradcam_heatmap(model, input_arr)
        if heatmap is not None:
            overlay = overlay_heatmap_on_image(pil_img, heatmap)
            st.image(overlay, caption="Grad-CAM heatmap overlay", use_column_width=True)
        else:
            st.warning("Grad-CAM unavailable for this model architecture.")

    # Logging the inference
    now = datetime.utcnow().isoformat()
    st.session_state["log"].append({"time": now, "label": label, "confidence": confidence, "prob": prob})
    st.write("Inference logged for this session.")

    # Option to download report CSV for this session
    if st.button("Download session results (CSV)"):
        import pandas as pd
        df = pd.DataFrame(st.session_state["log"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="smartdx_session_results.csv", mime="text/csv")

else:
    if model is None:
        st.warning("Model not loaded. Place malaria_model.keras inside the models/ folder.")
    else:
        st.info("Upload an image to analyze.")

# Footer: simple session log view
if st.sidebar.checkbox("Show session log"):
    st.sidebar.write(st.session_state.get("log", []))

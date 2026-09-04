import io
import streamlit as st
from PIL import Image
import numpy as np

from utils import load_model, predict_image

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="GeoRestore AI - Satellite Cloud Removal",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 GeoRestore AI")
st.subheader("Physics-Guided & Deep Residual Cloud Removal for Satellite Imagery")

st.markdown(
    """
    Restore crystal-clear, cloud-free imagery from any cloudy satellite capture.
    Works on unseen and arbitrary data without smudging clear land.
    """
)
st.markdown("---")

# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def get_model():
    return load_model()

try:
    model, device = get_model()
    device_name = "NVIDIA GPU (CUDA)" if device.type == "cuda" else "CPU"
    st.sidebar.success(f"⚡ Acceleration: **{device_name}**")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model, device = None, None

# ==========================================================
# Sidebar Settings
# ==========================================================
st.sidebar.title("⚙️ Restoration Controls")

mode_label = st.sidebar.selectbox(
    "Restoration Engine",
    options=[
        "🌟 Hybrid AI + Inpainting (Recommended)",
        "🌲 Contextual Terrain Inpainting",
        "🧠 Deep Residual U-Net"
    ],
    index=0,
    help="Select the restoration technique. Hybrid combines deep features with seamless terrain reconstruction."
)

mode_map = {
    "🌟 Hybrid AI + Inpainting (Recommended)": "hybrid",
    "🌲 Contextual Terrain Inpainting": "inpainting",
    "🧠 Deep Residual U-Net": "unet"
}
selected_mode = mode_map[mode_label]

sensitivity = st.sidebar.slider(
    "Cloud Sensitivity",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Higher values detect and remove lighter haze and thin clouds."
)

use_blending = st.sidebar.checkbox(
    "Ground Preservation Blending",
    value=True,
    help="Preserves 100% of clear ground pixels and seamlessly restores cloudy regions."
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **How it works:**
    1. **Robust Cloud Segmentation**: Distinguishes clouds from bright roofs and roads.
    2. **Contextual Terrain Reconstruction**: Reconstructs obscured ground context without color casts.
    3. **Edge-Preserving Blending**: 100% clear ground details are preserved.
    """
)

# ==========================================================
# Upload Image
# ==========================================================
uploaded_file = st.file_uploader(
    "Upload any cloudy satellite image (PNG, JPG, TIF)",
    type=["png", "jpg", "jpeg", "tif", "bmp"]
)

if uploaded_file is None:
    st.info("👆 Please upload a satellite image to begin cloud removal.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    width, height = image.size

    st.success(f"✅ Image loaded successfully! Dimensions: **{width} x {height}** pixels.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. Original Cloudy Image")
        st.image(image, use_container_width=True)

    if model is not None:
        if st.button("🚀 Restore Cloud-Free Image", type="primary"):
            with st.spinner("Processing cloud removal and terrain reconstruction..."):
                clean_output, cloud_mask = predict_image(
                    model=model,
                    device=device,
                    image=image,
                    apply_mask_blending=use_blending,
                    cloud_sensitivity=sensitivity,
                    mode=selected_mode
                )

            with col2:
                st.subheader("2. Detected Cloud Mask")
                # Normalize mask to colormap
                mask_display = (np.clip(cloud_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
                st.image(mask_display, use_container_width=True, caption="Cloud & Haze Confidence")

            with col3:
                st.subheader("3. Restored Cloud-Free Image")
                st.image(clean_output, use_container_width=True, caption="Cloud-Free Output")

                # Download button
                clean_pil = Image.fromarray(clean_output)
                buf = io.BytesIO()
                clean_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="💾 Download Cloud-Free Image",
                    data=byte_im,
                    file_name="georestore_cloud_free.png",
                    mime="image/png"
                )

            st.balloons()
            st.success("🎉 Restoration completed with seamless ground preservation!")

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path("fruit_classifier.keras")
LABELS_PATH = Path("class_names.json")

IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🥝",
    layout="wide"
)


# ==========================================
# HTML RENDERING HELPER
# ==========================================

def render_html(content):

    st.markdown(
        dedent(content).strip(),
        unsafe_allow_html=True
    )


# ==========================================
# CUSTOM CSS
# ==========================================

render_html("""
<style>

.stApp {
    background-color: #FFF8F2;
}

/* Main page */

.block-container {
    max-width: 1150px;
    padding-top: 2rem !important;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hide unnecessary elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background-color: transparent;
}

/* Main header */

.header-box {
    background: linear-gradient(110deg, #A30D43, #700D2B);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 35px;
    box-shadow: 0 5px 15px rgba(112, 13, 43, 0.12);
}

.header-title {
    color: white;
    font-size: 32px;
    font-weight: 700;
    margin: 0;
}

/* Section titles */

.section-title {
    color: #72102F;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Upload area */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1.5px dashed #C76B91;
    border-radius: 15px;
    padding: 15px;
}

/* Preview placeholder */

.preview-card {
    background-color: #F8E7EF;
    border-radius: 15px;
    height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #985475;
}

.preview-icon {
    font-size: 65px;
    margin-bottom: 10px;
}

.preview-text {
    font-size: 20px;
    font-weight: 600;
}

/* Uploaded image */

[data-testid="stImage"] img {
    height: 240px;
    width: 100%;
    object-fit: contain;
    border-radius: 15px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(110deg, #B30F4D, #720D2B);
    color: white;
    border: none;
    border-radius: 35px;
    font-size: 20px;
    font-weight: 700;
    min-height: 55px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: #5F0929;
    color: white;
    border: none;
}

/* Disabled button */

.stButton > button:disabled {
    background: linear-gradient(110deg, #A30D43, #700D2B) !important;
    color: white !important;
    opacity: 1 !important;
    border: none !important;
}

/* Prediction result */

.result-card {
    background-color: #F9EAF1;
    border: 1px solid #E9C9D7;
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    margin-top: 20px;
}

.result-label {
    color: #555555;
    font-size: 17px;
}

.result-name {
    color: #950D40;
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
}

.result-confidence {
    color: #35132E;
    font-size: 18px;
    font-weight: 600;
}

/* Information cards */

.info-card {
    background-color: #FFFFFF;
    border: 1px solid #E8C9D5;
    border-radius: 17px;
    padding: 18px 8px;
    text-align: center;
    min-height: 105px;
    box-shadow: 0 4px 10px rgba(114, 13, 43, 0.04);
}

.info-icon {
    font-size: 30px;
}

.info-heading {
    color: #850E3A;
    font-size: 17px;
    font-weight: 700;
}

.info-text {
    color: #353047;
    font-size: 15px;
}

/* Footer */

.footer-text {
    text-align: center;
    color: #9A8290;
    font-size: 13px;
    margin-top: 25px;
}

/* Divider */

hr {
    border-color: #DDBAC9;
    margin-top: 25px;
    margin-bottom: 25px;
}

/* Mobile */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .header-title {
        font-size: 25px;
    }

    .section-title {
        font-size: 18px;
    }

}

</style>
""")


# ==========================================
# HEADER
# ==========================================

render_html("""
<div class="header-box">
    <div class="header-title">
        🍊 Fruit Classifier
    </div>
</div>
""")


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError("fruit_classifier.keras is missing.")

    if not LABELS_PATH.exists():
        raise FileNotFoundError("class_names.json is missing.")

    model = tf.keras.models.load_model(MODEL_PATH)

    labels = json.loads(
        LABELS_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(labels, list):
        raise ValueError("Invalid class_names.json format.")

    if len(labels) != model.output_shape[-1]:
        raise ValueError("Model output and class labels do not match.")

    return model, labels


try:

    model, class_names = load_model()

except Exception as e:

    st.error(f"Model loading failed: {e}")

    st.stop()


# ==========================================
# IMAGE UPLOAD AND PREVIEW
# ==========================================

left, right = st.columns(2, gap="large")

uploaded = None
image = None


# ------------------------------------------
# UPLOAD SECTION
# ------------------------------------------

with left:

    render_html("""
    <div class="section-title">
        📤 Upload a Fruit Image
    </div>
    """)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.caption("JPG, JPEG, PNG and WEBP")


# ------------------------------------------
# PREVIEW SECTION
# ------------------------------------------

with right:

    render_html("""
    <div class="section-title">
        🖼️ Image Preview
    </div>
    """)

    if uploaded is not None:

        try:

            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                use_container_width=True
            )

        except Exception:

            st.error("Invalid image. Please upload another image.")

    else:

        render_html("""
        <div class="preview-card">
            <div class="preview-icon">🖼️</div>
            <div class="preview-text">Image Preview</div>
        </div>
        """)


# ==========================================
# CLASSIFY BUTTON
# ==========================================

st.write("")

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:

    classify = st.button(
        "✨  Classify",
        use_container_width=True,
        disabled=(uploaded is None or image is None)
    )


# ==========================================
# PREDICTION
# ==========================================

if classify and image is not None:

    try:
        with st.spinner("Analyzing your image..."):

            resized = image.resize(IMAGE_SIZE)

            arr = np.asarray(
                resized,
                dtype=np.float32
            )

            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)

            probabilities = model.predict(
                arr,
                verbose=0
            )[0]

            best_index = int(np.argmax(probabilities))

            predicted = class_names[best_index]

            confidence = float(probabilities[best_index])

        # Check whether the image is likely to be a fruit

        MIN_CONFIDENCE = 0.70

        st.markdown("---")

        if confidence < MIN_CONFIDENCE:

            st.warning(
                "⚠️ Not a fruit! Please upload a valid fruit image."
            )

        else:

            st.markdown("### 🍎 Prediction Result")

            st.success(f"Predicted Fruit: {predicted}")

            st.metric(
                label="Confidence",
                value=f"{confidence * 100:.2f}%"
            )

            with st.expander("View Class Probabilities"):

                sorted_results = sorted(
                    zip(class_names, probabilities),
                    key=lambda x: x[1],
                    reverse=True
                )

                for name, score in sorted_results:

                    st.write(
                        f"**{name}** — {score * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0, 1))
                    )

    except Exception as e:

        st.error(f"Prediction failed: {e}")

# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()

c1, c2, c3 = st.columns(3, gap="medium")


with c1:

    render_html("""
    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-heading">Model</div>
        <div class="info-text">MobileNetV2</div>
    </div>
    """)


with c2:

    render_html("""
    <div class="info-card">
        <div class="info-icon">📚</div>
        <div class="info-heading">Dataset</div>
        <div class="info-text">Fruits-360</div>
    </div>
    """)


with c3:

    render_html("""
    <div class="info-card">
        <div class="info-icon">⚙️</div>
        <div class="info-heading">Approach</div>
        <div class="info-text">Transfer Learning</div>
    </div>
    """)


# ==========================================
# FOOTER
# ==========================================

render_html("""
<div class="footer-text">
    Fruit Classifier | Deep Learning Project
</div>
""")

import json
from pathlib import Path

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
    page_icon="🍎",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #FFF8F2;
}

/* Main container */

.block-container {
    max-width: 1200px;
    padding-top: 2rem !important;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Hide unnecessary Streamlit elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Header */

.header-box {
    background: linear-gradient(110deg, #A20E43, #720D2B);
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 35px;
    text-align: center;
    box-shadow: 0 6px 15px rgba(114, 13, 43, 0.12);
}

.header-title {
    color: white;
    font-size: 32px;
    font-weight: 700;
    margin: 0;
    text-align: center;
    letter-spacing: 0.5px;
}

/* Section headings */

.section-title {
    color: #72102F;
    font-size: 21px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* Upload card */

.upload-card {
    background-color: #FFFFFF;
    border: 1px solid #EBD5DC;
    border-radius: 18px;
    padding: 22px;
    min-height: 240px;
    box-shadow: 0 4px 12px rgba(114, 13, 43, 0.05);
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #FFF9FC;
    border: 1.5px dashed #C86B91;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background-color: transparent;
}

/* Preview card */

.preview-card {
    background-color: #F8E7EF;
    border-radius: 15px;
    min-height: 240px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #985475;
    padding: 15px;
}

.preview-icon {
    font-size: 65px;
    margin-bottom: 8px;
}

.preview-text {
    font-size: 20px;
    font-weight: 600;
}

/* Uploaded image */

[data-testid="stImage"] {
    background-color: #F8E7EF;
    border-radius: 15px;
    padding: 8px;
}

[data-testid="stImage"] img {
    max-height: 240px;
    object-fit: contain;
    border-radius: 12px;
}

/* Classify button */

.stButton > button {
    background: linear-gradient(110deg, #B30F4D, #720D2B);
    color: white;
    border: none;
    border-radius: 35px;
    font-size: 21px;
    font-weight: 700;
    min-height: 58px;
    box-shadow: 0 5px 12px rgba(114, 13, 43, 0.15);
    transition: 0.2s;
}

.stButton > button:hover {
    background: #5F0929;
    color: white;
    border: none;
}

/* Disabled button */

.stButton > button:disabled {
    background: #D9B5C4;
    color: white;
    box-shadow: none;
}

/* Result card */

.result-card {
    background-color: #F9EAF1;
    border: 1px solid #E9C9D7;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-top: 15px;
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
    padding: 20px 10px;
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

/* Mobile layout */

@media (max-width: 768px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem !important;
    }

    .header-title {
        font-size: 25px;
    }

    .section-title {
        font-size: 18px;
    }

}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="header-box">
    <div class="header-title">
        🍎 Fruit Classifier
    </div>
</div>
""", unsafe_allow_html=True)


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
        raise ValueError("class_names.json must contain a list of class names.")

    if len(labels) != model.output_shape[-1]:
        raise ValueError("Number of class names does not match model outputs.")

    return model, labels


try:
    model, class_names = load_model()

except Exception as e:

    st.error(f"Unable to load the trained model: {e}")

    st.stop()


# ==========================================
# IMAGE UPLOAD AND PREVIEW
# ==========================================

left, right = st.columns(2, gap="large")


# ------------------------------------------
# LEFT: UPLOAD SECTION
# ------------------------------------------

with left:

    st.markdown("""
    <div class="section-title">
        📤 Upload a Fruit Image
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")


# ------------------------------------------
# RIGHT: IMAGE PREVIEW
# ------------------------------------------

image = None

with right:

    st.markdown("""
    <div class="section-title">
        🖼️ Image Preview
    </div>
    """, unsafe_allow_html=True)

    if uploaded:

        try:

            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                use_container_width=True
            )

        except Exception:

            st.error("Unable to read this image. Please upload another image.")

    else:

        st.markdown("""
        <div class="preview-card">

            <div class="preview-icon">🖼️</div>

            <div class="preview-text">
                Image Preview
            </div>

        </div>
        """, unsafe_allow_html=True)


# ==========================================
# CLASSIFICATION BUTTON
# ==========================================

st.write("")

button_col1, button_col2, button_col3 = st.columns([1, 1.5, 1])

with button_col2:

    classify = st.button(
        "✨  Classify",
        use_container_width=True,
        disabled=(uploaded is None or image is None)
    )


# ==========================================
# PREDICTION
# ==========================================

if classify and uploaded and image is not None:

    try:

        with st.spinner("Analyzing your image..."):

            # Resize image

            resized = image.resize(IMAGE_SIZE)

            # Convert image to array

            arr = np.asarray(
                resized,
                dtype=np.float32
            )

            # MobileNetV2 preprocessing

            arr = preprocess_input(arr)

            # Add batch dimension

            arr = np.expand_dims(arr, axis=0)

            # Model prediction

            probabilities = model.predict(
                arr,
                verbose=0
            )[0]

            # Find predicted class

            best_index = int(np.argmax(probabilities))

            predicted = class_names[best_index]

            confidence = float(probabilities[best_index]) * 100

        # Display result

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                Predicted Fruit
            </div>

            <div class="result-name">
                🍎 {predicted}
            </div>

            <div class="result-confidence">
                Confidence: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        # Display probabilities

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


# Model

with c1:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">🧠</div>

        <div class="info-heading">
            Model
        </div>

        <div class="info-text">
            MobileNetV2
        </div>

    </div>
    """, unsafe_allow_html=True)


# Dataset

with c2:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">📚</div>

        <div class="info-heading">
            Dataset
        </div>

        <div class="info-text">
            Fruits-360
        </div>

    </div>
    """, unsafe_allow_html=True)


# Approach

with c3:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">⚙️</div>

        <div class="info-heading">
            Approach
        </div>

        <div class="info-text">
            Transfer Learning
        </div>

    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer-text">
    Fruit Classifier | Deep Learning Project
</div>
""", unsafe_allow_html=True)

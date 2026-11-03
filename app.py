import json
from pathlib import Path
from html import escape

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

/* Background */

.stApp {
    background-color: #FFF9F2;
}

/* Hide Streamlit header */

[data-testid="stHeader"] {
    height: 0;
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none;
}

#MainMenu,
footer {
    visibility: hidden;
}

/* Main container */

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem !important;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Top maroon bar */

.header-box {
    background-color: #720D2B;
    border-radius: 12px;
    padding: 19px 20px;
    text-align: center;
    margin-bottom: 35px;
    width: 100%;
    box-sizing: border-box;
}

.header-title {
    color: white !important;
    font-size: 27px;
    font-weight: 700;
    margin: 0;
    padding: 0;
}

/* Section headings */

.section-title {
    color: #35132E;
    font-size: 21px;
    font-weight: 650;
    margin-bottom: 15px;
}

/* Upload box */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #B99AA7;
    border-radius: 12px;
    padding: 13px;
    min-height: 110px;
}

/* File uploader text */

[data-testid="stFileUploader"] section {
    background-color: #F7F4F5;
    border-radius: 9px;
}

/* Preview image */

[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

[data-testid="stImage"] img {
    max-height: 220px;
    object-fit: contain;
    border-radius: 12px;
}

/* Empty preview box */

.preview-box {
    background-color: #F3EBED;
    height: 205px;
    border-radius: 12px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #8A7880;
    font-size: 15px;
    text-align: center;
}

/* Supported formats */

.upload-caption {
    color: #888888;
    font-size: 13px;
    margin-top: 8px;
}

/* Classification button */

.stButton > button {
    background-color: #850E3A;
    color: white;
    border: none;
    border-radius: 9px;
    font-size: 17px;
    font-weight: 600;
    min-height: 46px;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #5F0929;
    color: white;
    border: none;
}

/* Prediction card */

.result-card {
    background-color: #F4EAF0;
    border: 1px solid #E5CCD8;
    border-radius: 13px;
    padding: 16px;
    text-align: center;
    margin-top: 18px;
}

.result-label {
    color: #666666;
    font-size: 15px;
}

.result-name {
    color: #850E3A;
    font-size: 28px;
    font-weight: bold;
    margin: 5px 0;
}

.result-confidence {
    color: #35132E;
    font-size: 16px;
    font-weight: 600;
}

/* Expander */

[data-testid="stExpander"] {
    border: 1px solid #E5CCD8;
    border-radius: 10px;
    background-color: white;
    margin-top: 10px;
}

/* Reduce vertical spacing */

div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

/* Responsive design */

@media (max-width: 650px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .header-title {
        font-size: 23px;
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
    <p class="header-title">🍎 Fruit Classifier</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    labels = json.loads(
        LABELS_PATH.read_text(encoding="utf-8")
    )

    return model, labels


model, class_names = load_model()


if model is None:

    st.error(
        "Trained model files are missing. "
        "Please add fruit_classifier.keras and "
        "class_names.json to the application folder."
    )

    st.stop()


# ==========================================
# IMAGE UPLOAD AND PREVIEW
# ==========================================

left, right = st.columns(2, gap="large")


# Upload section

with left:

    st.markdown(
        '<div class="section-title">📤 Upload Image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose a fruit image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="upload-caption">'
        'Supported formats: JPG, PNG, WEBP'
        '</div>',
        unsafe_allow_html=True
    )


# Preview section

with right:

    st.markdown(
        '<div class="section-title">🖼️ Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded:

        try:
            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                use_container_width=True
            )

        except Exception:
            st.error("Unable to open this image.")
            st.stop()

    else:

        st.markdown("""
        <div class="preview-box">
            Your image will appear here
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# CLASSIFICATION BUTTON
# ==========================================

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    classify = st.button(
        "✨  Classify Fruit",
        use_container_width=True,
        disabled=uploaded is None
    )


# ==========================================
# PREDICTION
# ==========================================

if classify and uploaded:

    try:

        with st.spinner("Analyzing image..."):

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

            if best_index >= len(class_names):
                raise ValueError(
                    "Model output does not match the class labels."
                )

            predicted = str(class_names[best_index])

            confidence = (
                float(probabilities[best_index]) * 100
            )

        # Prediction result

        st.markdown(f"""
        <div class="result-card">

            <div class="result-label">
                Predicted Fruit
            </div>

            <div class="result-name">
                🍎 {escape(predicted)}
            </div>

            <div class="result-confidence">
                Confidence: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        # Class probabilities

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

                st.progress(float(score))

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div style="
    text-align:center;
    color:#999999;
    font-size:12px;
    margin-top:25px;
">
    Fruit Classifier | Deep Learning Project
</div>
""", unsafe_allow_html=True)

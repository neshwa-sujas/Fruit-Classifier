import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------- CONFIGURATION ----------------

MODEL_PATH = Path("fruit_classifier.keras")
LABELS_PATH = Path("class_names.json")
IMAGE_SIZE = (160, 160)

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #FFF9F0;
}

/* Remove default top padding */

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Navigation bar */

.navbar {
    background-color: #720D2B;
    padding: 18px 30px;
    border-radius: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 35px;
}

.nav-title {
    color: white;
    font-size: 27px;
    font-weight: bold;
}

.nav-icon {
    font-size: 28px;
}

/* Main heading */

.main-heading {
    text-align: center;
    color: #35132E;
    font-size: 43px;
    font-weight: bold;
    margin-bottom: 5px;
}

.sub-heading {
    text-align: center;
    color: #666666;
    font-size: 19px;
    margin-bottom: 35px;
}

/* Content card */

.main-card {
    background-color: white;
    border-radius: 18px;
    padding: 30px;
    border: 1px solid #E8DDD6;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.06);
}

/* Upload heading */

.upload-heading {
    color: #35132E;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 15px;
}

/* Prediction card */

.result-card {
    background-color: #F2EAF5;
    padding: 22px;
    border-radius: 15px;
    text-align: center;
    margin-top: 15px;
}

.result-title {
    color: #35132E;
    font-size: 23px;
    font-weight: bold;
}

.result-fruit {
    color: #8A1239;
    font-size: 30px;
    font-weight: bold;
}

.confidence {
    color: #35132E;
    font-size: 20px;
    font-weight: bold;
    margin-top: 10px;
}

/* Buttons */

.stButton > button {
    background-color: #850E3A;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background-color: #5F0929;
    color: white;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #FFFCFC;
    border: 1px dashed #A99AAE;
    border-radius: 12px;
    padding: 15px;
}

/* Information cards */

.info-container {
    background-color: white;
    border: 1px solid #E5DCE2;
    border-radius: 12px;
    padding: 20px 10px;
    text-align: center;
    margin-top: 25px;
}

.info-title {
    color: #35132E;
    font-size: 20px;
    font-weight: bold;
}

.info-value {
    color: #555555;
    font-size: 17px;
}

/* Footer */

.footer {
    text-align: center;
    color: #777777;
    font-size: 14px;
    margin-top: 35px;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- NAVIGATION BAR ----------------

st.markdown("""
<div class="navbar">
    <div class="nav-title">
        🍎 Fruit Classifier
    </div>
    <div style="color:white; font-size:16px;">
        Home &nbsp;&nbsp;&nbsp; About &nbsp;&nbsp;&nbsp; Dataset
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- MAIN HEADING ----------------

st.markdown("""
<div class="main-heading">
    Fruit Classification
</div>

<div class="sub-heading">
    Identify fruits instantly using Deep Learning
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD MODEL ----------------

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
        "class_names.json to the application."
    )

    st.stop()


# ---------------- IMAGE UPLOAD SECTION ----------------

left, right = st.columns(2, gap="large")

with left:

    st.markdown(
        '<div class="upload-heading">📤 Upload a Fruit Image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.caption(
        "Supported formats: JPG, JPEG, PNG, WEBP"
    )


with right:

    st.markdown(
        '<div class="upload-heading">🖼️ Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            use_container_width=True
        )

    else:

        st.markdown("""
        <div style="
            background-color:#F8F3F4;
            height:240px;
            border-radius:12px;
            display:flex;
            justify-content:center;
            align-items:center;
            color:#777777;
            font-size:18px;
            text-align:center;
        ">
            Your uploaded image will appear here
        </div>
        """, unsafe_allow_html=True)


# ---------------- CLASSIFICATION ----------------

st.write("")

if uploaded:

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        classify = st.button(
            "✨ Classify Fruit",
            use_container_width=True
        )

    if classify:

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

            confidence = (
                float(probabilities[best_index]) * 100
            )

        st.markdown(f"""
        <div class="result-card">

            <div class="result-title">
                Predicted Fruit
            </div>

            <div class="result-fruit">
                🍎 {predicted}
            </div>

            <div class="confidence">
                Confidence: {confidence:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

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


# ---------------- MODEL INFORMATION ----------------

st.write("")

st.markdown("---")

st.markdown(
    "<h3 style='text-align:center; color:#35132E;'>"
    "Model Information</h3>",
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown("""
    <div class="info-container">
        <div style="font-size:30px;">🧠</div>
        <div class="info-title">Model</div>
        <div class="info-value">MobileNetV2</div>
    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-container">
        <div style="font-size:30px;">🗂️</div>
        <div class="info-title">Dataset</div>
        <div class="info-value">Fruits-360</div>
    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-container">
        <div style="font-size:30px;">⚙️</div>
        <div class="info-title">Approach</div>
        <div class="info-value">Transfer Learning</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
    Fruit Classifier | Deep Learning Project
    <br>
    Powered by TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)

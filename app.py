
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
    page_icon="🍎",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Page background */

.stApp {
    background-color: #FFF9F2;
}

/* Main container */

.block-container {
    max-width: 1100px;
    padding-top: 1rem;
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
    background-color: #720D2B;
    padding: 15px 25px;
    border-radius: 12px;
    margin-bottom: 20px;
}

.header-title {
    color: white;
    font-size: 27px;
    font-weight: 700;
    margin: 0;
}

/* Main title */

.main-title {
    color: #35132E;
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 3px;
}

.subtitle {
    color: #777777;
    text-align: center;
    font-size: 16px;
    margin-bottom: 22px;
}

/* Section headings */

.section-title {
    color: #35132E;
    font-size: 21px;
    font-weight: 650;
    margin-bottom: 12px;
}

/* Upload section */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #B99AA7;
    border-radius: 12px;
    padding: 12px;
}

/* Image preview */

[data-testid="stImage"] img {
    max-height: 230px;
    object-fit: contain;
    border-radius: 10px;
}

/* Buttons */

.stButton > button {
    background-color: #850E3A;
    color: white;
    border: none;
    border-radius: 9px;
    font-size: 17px;
    font-weight: 600;
    min-height: 45px;
}

.stButton > button:hover {
    background-color: #5F0929;
    color: white;
    border: none;
}

/* Result card */

.result-card {
    background-color: #F4EAF0;
    border: 1px solid #E5CCD8;
    border-radius: 13px;
    padding: 17px;
    text-align: center;
    margin-top: 15px;
}

.result-label {
    color: #555555;
    font-size: 16px;
}

.result-name {
    color: #850E3A;
    font-size: 29px;
    font-weight: bold;
    margin: 5px 0;
}

.result-confidence {
    color: #35132E;
    font-size: 17px;
    font-weight: 600;
}

/* Information cards */

.info-card {
    background-color: white;
    border: 1px solid #E8DDE2;
    border-radius: 10px;
    padding: 12px 5px;
    text-align: center;
}

.info-icon {
    font-size: 23px;
}

.info-heading {
    color: #35132E;
    font-size: 15px;
    font-weight: bold;
}

.info-text {
    color: #777777;
    font-size: 13px;
}

/* Footer */

.footer-text {
    text-align: center;
    color: #888888;
    font-size: 12px;
    margin-top: 18px;
}

/* Reduce vertical spacing */

div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    dedent("""
    <div class="header-box">
        <p class="header-title">🍎 Fruit Classifier</p>
    </div>
    """),
    unsafe_allow_html=True
)


# ==========================================
# MAIN HEADING
# ==========================================

st.markdown(
    dedent("""
    <div class="main-title">Fruit Classification</div>
    <div class="subtitle">
        Identify fruits instantly using Deep Learning
    </div>
    """),
    unsafe_allow_html=True
)


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


with left:

    st.markdown(
        '<div class="section-title">📤 Upload a Fruit Image</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.caption("Supported formats: JPG, JPEG, PNG, WEBP")


with right:

    st.markdown(
        '<div class="section-title">🖼️ Image Preview</div>',
        unsafe_allow_html=True
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            width=350
        )

    else:

        st.markdown(
            dedent("""
            <div style="
                background-color: #F3EBED;
                height: 190px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                text-align: center;
                color: #777777;
                font-size: 16px;
            ">
                Your uploaded image will appear here
            </div>
            """),
            unsafe_allow_html=True
        )


# ==========================================
# CLASSIFICATION BUTTON
# ==========================================

st.write("")

button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

with button_col2:

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

        # Display prediction

        st.markdown(
            dedent(f"""
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
            """),
            unsafe_allow_html=True
        )

        # Display class probabilities

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
# MODEL INFORMATION
# ==========================================

st.divider()

st.markdown(
    '<h3 style="text-align:center; color:#35132E;">'
    'Model Information</h3>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)


with c1:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="info-icon">🧠</div>
            <div class="info-heading">Model</div>
            <div class="info-text">MobileNetV2</div>
        </div>
        """),
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="info-icon">🗂️</div>
            <div class="info-heading">Dataset</div>
            <div class="info-text">Fruits-360</div>
        </div>
        """),
        unsafe_allow_html=True
    )


with c3:

    st.markdown(
        dedent("""
        <div class="info-card">
            <div class="info-icon">⚙️</div>
            <div class="info-heading">Approach</div>
            <div class="info-text">Transfer Learning</div>
        </div>
        """),
        unsafe_allow_html=True
    )


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer-text">'
    'Fruit Classifier | Deep Learning Project'
    '</div>',
    unsafe_allow_html=True
)

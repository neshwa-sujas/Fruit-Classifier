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

/* Hide default Streamlit header */

header[data-testid="stHeader"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main page */

.stApp {
    background-color: #FFF9F0;
}

.block-container {
    padding-top: 0.8rem;
    padding-bottom: 0.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1250px;
}

/* Navigation bar */

.navbar {
    background-color: #720D2B;
    padding: 13px 28px;
    border-radius: 0 0 12px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 48px;
    box-sizing: border-box;
    margin-bottom: 18px;
}

.nav-title {
    color: white;
    font-size: 25px;
    font-weight: bold;
    line-height: 1.3;
    white-space: nowrap;
}

.nav-links {
    color: white;
    font-size: 15px;
    font-weight: 500;
}

/* Main heading */

.main-heading {
    text-align: center;
    color: #35132E;
    font-size: 36px;
    font-weight: bold;
    margin-top: 5px;
    margin-bottom: 3px;
}

.sub-heading {
    text-align: center;
    color: #666666;
    font-size: 16px;
    margin-bottom: 15px;
}

/* Section headings */

.section-heading {
    text-align: center;
    color: #35132E;
    font-size: 21px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Upload section */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 1px dashed #A99AAE;
    border-radius: 12px;
    padding: 12px;
}

/* Image preview */

[data-testid="stImage"] {
    display: flex;
    justify-content: center;
}

[data-testid="stImage"] img {
    max-height: 235px;
    object-fit: contain;
    border-radius: 10px;
}

/* Buttons */

.stButton > button {
    background-color: #850E3A;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 17px;
    font-weight: bold;
    padding: 10px;
    min-height: 43px;
    transition: 0.2s;
}

.stButton > button:hover {
    background-color: #5F0929;
    color: white;
    border: none;
}

/* Prediction card */

.result-card {
    background-color: #F2EAF5;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    margin-top: 8px;
    border: 1px solid #E5DCE2;
}

.result-title {
    color: #35132E;
    font-size: 17px;
    font-weight: bold;
}

.result-fruit {
    color: #850E3A;
    font-size: 26px;
    font-weight: bold;
}

.confidence {
    color: #35132E;
    font-size: 17px;
    font-weight: bold;
    margin-top: 4px;
}

/* Model information */

.info-container {
    background-color: white;
    border: 1px solid #E5DCE2;
    border-radius: 10px;
    padding: 9px;
    text-align: center;
}

.info-icon {
    font-size: 23px;
}

.info-title {
    color: #35132E;
    font-size: 16px;
    font-weight: bold;
}

.info-value {
    color: #555555;
    font-size: 14px;
}

/* Footer */

.footer {
    text-align: center;
    color: #777777;
    font-size: 12px;
    margin-top: 12px;
}

/* Reduce vertical gaps */

div[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# NAVIGATION BAR
# ==========================================

st.markdown("""
<div class="navbar">

    <div class="nav-title">
        🍎 Fruit Classifier
    </div>

    <div class="nav-links">
        Home &nbsp;&nbsp;&nbsp;
        About &nbsp;&nbsp;&nbsp;
        Dataset
    </div>

</div>
""", unsafe_allow_html=True)


# ==========================================
# MAIN HEADING
# ==========================================

st.markdown("""
<div class="main-heading">
    Fruit Classification
</div>

<div class="sub-heading">
    Identify fruits instantly using Deep Learning
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
        "class_names.json to the application."
    )

    st.stop()


# ==========================================
# UPLOAD AND PREVIEW
# ==========================================

left, right = st.columns(2, gap="large")


with left:

    st.markdown("""
    <div class="section-heading">
        📤 Upload a Fruit Image
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.caption(
        "Supported formats: JPG, JPEG, PNG, WEBP"
    )


with right:

    st.markdown("""
    <div class="section-heading">
        🖼️ Image Preview
    </div>
    """, unsafe_allow_html=True)

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            width=360
        )

    else:

        st.markdown("""
        <div style="
            background-color:#F5EFF1;
            height:180px;
            border-radius:12px;
            display:flex;
            justify-content:center;
            align-items:center;
            color:#777777;
            font-size:17px;
            text-align:center;
        ">
            Your uploaded image will appear here
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# CLASSIFICATION
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    classify = st.button(
        "✨  Classify Fruit",
        use_container_width=True,
        disabled=uploaded is None
    )


if classify and uploaded:

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


    # Prediction result

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


# ==========================================
# MODEL INFORMATION
# ==========================================

st.markdown("---")

c1, c2, c3 = st.columns(3)


with c1:

    st.markdown("""
    <div class="info-container">

        <div class="info-icon">🧠</div>

        <div class="info-title">
            Model
        </div>

        <div class="info-value">
            MobileNetV2
        </div>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-container">

        <div class="info-icon">🗂️</div>

        <div class="info-title">
            Dataset
        </div>

        <div class="info-value">
            Fruits-360
        </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-container">

        <div class="info-icon">⚙️</div>

        <div class="info-title">
            Approach
        </div>

        <div class="info-value">
            Transfer Learning
        </div>

    </div>
    """, unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    Fruit Classifier | Deep Learning Project
</div>
""", unsafe_allow_html=True)

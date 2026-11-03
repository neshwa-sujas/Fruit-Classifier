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

/* PAGE BACKGROUND */

.stApp {
    background: linear-gradient(
        135deg,
        #FFF8F1,
        #FFFDFB,
        #FFF2F3
    );
}

/* HIDE STREAMLIT ELEMENTS */

[data-testid="stHeader"] {
    display: none;
}

[data-testid="stToolbar"] {
    display: none;
}

#MainMenu,
footer {
    visibility: hidden;
}

/* MAIN CONTAINER */

.block-container {
    max-width: 1250px;
    padding-top: 2rem !important;
    padding-bottom: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* =================================
   HEADER
================================= */

.header-box {
    background: linear-gradient(
        110deg,
        #A20D43,
        #850B38,
        #670B30
    );

    border-radius: 27px;

    min-height: 145px;

    display: flex;
    justify-content: center;
    align-items: center;

    text-align: center;

    margin-bottom: 35px;

    box-shadow: 0 12px 25px rgba(133, 11, 56, 0.15);

    position: relative;

    overflow: hidden;
}

.header-box::before {
    content: "🍃";
    position: absolute;
    left: 35px;
    top: 35px;
    font-size: 48px;
    opacity: 0.35;
}

.header-box::after {
    content: "🍃";
    position: absolute;
    right: 35px;
    top: 35px;
    font-size: 48px;
    opacity: 0.35;
}

.header-title {
    color: white !important;
    font-size: 48px;
    font-weight: 750;
    margin: 0;
    letter-spacing: 0.5px;
}

/* =================================
   DECORATIVE FRUITS
================================= */

.fruit-decoration {
    text-align: left;
    font-size: 35px;
    margin-top: -15px;
    margin-bottom: 15px;
}

/* =================================
   SECTION HEADINGS
================================= */

.section-title {
    color: #780D35;
    font-size: 22px;
    font-weight: 750;
    text-align: center;
    margin-bottom: 18px;
}

/* =================================
   UPLOAD CARD
================================= */

.upload-card {
    background: rgba(255,255,255,0.85);
    border: 1px solid #EBD3DB;
    border-radius: 25px;

    padding: 25px 22px;

    min-height: 290px;

    box-shadow: 0 8px 20px rgba(120, 13, 53, 0.06);
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {
    background: #FFF9FC;
    border: 2px dashed #D78BAE;
    border-radius: 19px;
    padding: 18px;
    min-height: 190px;
    display: flex;
    justify-content: center;
}

[data-testid="stFileUploader"] section {
    background: transparent;
    border: none;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(
        110deg,
        #C30D50,
        #850B38
    );
    color: white;
    border: none;
    border-radius: 30px;
}

[data-testid="stFileUploader"] button:hover {
    color: white;
    border: none;
    background: #690A30;
}

/* =================================
   PREVIEW CARD
================================= */

.preview-card {
    background: linear-gradient(
        135deg,
        #FBE5F0,
        #F7DDEA
    );

    border-radius: 20px;

    height: 220px;

    display: flex;
    flex-direction: column;

    justify-content: center;
    align-items: center;

    text-align: center;

    color: #965B7C;
}

.preview-icon {
    font-size: 65px;
    margin-bottom: 8px;
}

.preview-text {
    font-size: 19px;
    font-weight: 650;
}

/* PREVIEW IMAGE */

[data-testid="stImage"] {
    background: #FBE5F0;
    border-radius: 20px;
    padding: 10px;
    height: 220px;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}

[data-testid="stImage"] img {
    max-height: 200px;
    object-fit: contain;
    border-radius: 12px;
}

/* =================================
   CLASSIFY BUTTON
================================= */

.stButton {
    display: flex;
    justify-content: center;
}

.stButton > button {

    background: linear-gradient(
        110deg,
        #C30D50,
        #850B38,
        #690A30
    );

    color: white;

    border: none;
    border-radius: 35px;

    min-height: 65px;

    font-size: 22px;
    font-weight: 750;

    box-shadow: 0 8px 18px rgba(133, 11, 56, 0.18);

    transition: 0.25s;
}

.stButton > button:hover {
    background: linear-gradient(
        110deg,
        #A20D43,
        #670B30
    );

    color: white;
    border: none;

    transform: translateY(-2px);
}

/* =================================
   PREDICTION RESULT
================================= */

.result-card {
    background: linear-gradient(
        135deg,
        #FBE5F0,
        #F5DCE8
    );

    border: 1px solid #E8C5D5;
    border-radius: 20px;

    padding: 22px;

    text-align: center;

    margin-top: 20px;
}

.result-label {
    color: #735568;
    font-size: 17px;
}

.result-name {
    color: #850B38;
    font-size: 32px;
    font-weight: 750;
    margin: 8px 0;
}

.result-confidence {
    color: #35132E;
    font-size: 17px;
    font-weight: 650;
}

/* =================================
   DIVIDER
================================= */

.custom-divider {
    height: 2px;

    background: linear-gradient(
        to right,
        transparent,
        #D58AA9,
        #850B38,
        #D58AA9,
        transparent
    );

    margin-top: 35px;
    margin-bottom: 30px;
}

/* =================================
   INFORMATION CARDS
================================= */

.info-card {
    background: rgba(255,255,255,0.75);

    border: 1px solid #E5C5D0;

    border-radius: 22px;

    min-height: 105px;

    display: flex;

    justify-content: center;
    align-items: center;

    gap: 17px;

    padding: 12px;

    box-shadow: 0 5px 15px rgba(120, 13, 53, 0.05);
}

.info-icon {
    font-size: 43px;
}

.info-heading {
    color: #850B38;
    font-size: 19px;
    font-weight: 750;
}

.info-text {
    color: #25243C;
    font-size: 17px;
    margin-top: 4px;
}

/* =================================
   FOOTER
================================= */

.footer {
    text-align: center;
    color: #987B89;
    font-size: 13px;
    margin-top: 35px;
    padding-bottom: 12px;
}

/* =================================
   RESPONSIVE DESIGN
================================= */

@media (max-width: 700px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .header-box {
        min-height: 100px;
    }

    .header-title {
        font-size: 30px;
    }

    .section-title {
        font-size: 18px;
    }

    .info-card {
        margin-bottom: 10px;
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
# DECORATIVE FRUITS
# ==========================================

st.markdown("""
<div class="fruit-decoration">
    🍓 🍊 🍃
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL
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
# UPLOAD AND PREVIEW
# ==========================================

left, right = st.columns(2, gap="large")


# UPLOAD

with left:

    st.markdown("""
    <div class="section-title">
        Upload a Fruit Image
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    st.caption("Supported formats: JPG, PNG, WEBP")


# PREVIEW

with right:

    st.markdown("""
    <div class="section-title">
        Image Preview
    </div>
    """, unsafe_allow_html=True)

    image = None

    if uploaded:

        try:

            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                use_container_width=True
            )

        except Exception:

            st.error("Invalid image file.")
            st.stop()

    else:

        st.markdown("""
        <div class="preview-card">

            <div class="preview-icon">
                🖼️
            </div>

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
        disabled=uploaded is None
    )


# ==========================================
# PREDICTION
# ==========================================

if classify and uploaded and image is not None:

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

            if best_index >= len(class_names):
                raise ValueError("Model labels do not match its output.")

            predicted = str(class_names[best_index])

            confidence = float(probabilities[best_index]) * 100

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
# DIVIDER
# ==========================================

st.markdown("""
<div class="custom-divider"></div>
""", unsafe_allow_html=True)


# ==========================================
# MODEL INFORMATION
# ==========================================

c1, c2, c3 = st.columns(3, gap="medium")


with c1:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">🧠</div>

        <div>
            <div class="info-heading">Model</div>
            <div class="info-text">MobileNetV2</div>
        </div>

    </div>
    """, unsafe_allow_html=True)


with c2:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">📚</div>

        <div>
            <div class="info-heading">Dataset</div>
            <div class="info-text">Fruits-360</div>
        </div>

    </div>
    """, unsafe_allow_html=True)


with c3:

    st.markdown("""
    <div class="info-card">

        <div class="info-icon">⚙️</div>

        <div>
            <div class="info-heading">Approach</div>
            <div class="info-text">Transfer Learning</div>
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

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = Path("fruit_classifier.keras")
LABELS_PATH = Path("class_names.json")
IMAGE_SIZE = (160, 160)

st.set_page_config(page_title="Fruit AI Classifier", page_icon="🍎", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return model, labels

st.title("🍎 Fruit AI Classifier")
st.write("Upload a picture of an apple, banana, orange, mango, grapes, or strawberry.")
st.caption("This is a student demonstration model. Predictions may be incorrect for unusual images.")

model, class_names = load_model()

if model is None:
    st.error("The trained model files are missing. Complete the Google Colab training notebook and add fruit_classifier.keras and class_names.json to this app folder.")
    st.stop()

uploaded = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify fruit", type="primary", use_container_width=True):
        resized = image.resize(IMAGE_SIZE)
        arr = np.asarray(resized, dtype=np.float32)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        probabilities = model.predict(arr, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        predicted = class_names[best_index]
        confidence = float(probabilities[best_index]) * 100

        st.success(f"Prediction: **{predicted}**")
        st.metric("Model confidence", f"{confidence:.1f}%")

        st.subheader("Class probabilities")
        for name, score in sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True):
            st.write(f"{name}: {score * 100:.1f}%")
            st.progress(float(score))

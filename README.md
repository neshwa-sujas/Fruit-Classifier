# 🍎 Fruit Classifier

## Project Overview

Fruit Classifier is a web-based application that identifies different types of fruits from uploaded images using Deep Learning.

The application uses a pre-trained MobileNetV2 model with Transfer Learning. The model was trained using the Fruits-360 dataset in Google Colab. A simple and user-friendly interface was developed using Streamlit.

## 🚀 Live Application

**Try the Fruit Classifier here:**

https://fruit-classifier-neshwa.streamlit.app/

## Features

* Upload a fruit image.
* Preview the uploaded image.
* Predict the fruit category using a trained deep learning model.
* Display the predicted fruit name.

## Technologies Used

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Pre-trained Model:** MobileNetV2
* **Technique:** Transfer Learning
* **Dataset:** Fruits-360
* **Web Framework:** Streamlit
* **Development Environment:** Google Colab
* **Version Control:** GitHub

## Model Details

* **Model:** MobileNetV2
* **Input Image Size:** 160 × 160 pixels
* **Preprocessing:** MobileNetV2 preprocessing
* **Model Format:** `.keras`
* **Training Dataset:** Fruits-360

## Project Structure

```text
Fruit-Classifier/
│
├── app.py
├── fruit_classifier.keras
├── class_names.json
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd Fruit-Classifier
```

### 3. Install the Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## Deployment

The application is deployed using **Streamlit Community Cloud** and connected to the GitHub repository.

Users can access the application through the live URL provided above without installing Python or running the project locally.

## Conclusion

Fruit Classifier demonstrates how Deep Learning and Transfer Learning can be used to identify fruits from images. By combining MobileNetV2 and Streamlit, the project provides a simple and interactive fruit classification application.

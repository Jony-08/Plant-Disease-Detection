
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Disease categories - same order as training
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# Application heading
st.title("🌿 Plant Disease Detection")
st.write("Upload a plant leaf image to identify its disease.")

st.info("This AI model supports 15 Pepper, Potato and Tomato categories.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf", width=300)

    if st.button("Predict Disease"):

        # Prepare image
        image = image.resize((128, 128))

        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)

        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0])) * 100

        disease_name = class_names[predicted_index]

        # Display result
        st.success("Prediction Completed!")

        st.subheader("Detection Result")

        st.write("**Predicted Category:**", disease_name)

        st.write(f"**Model Confidence:** {confidence:.2f}%")

        st.progress(confidence / 100)

        if "healthy" in disease_name.lower():
            st.success("The model classified this leaf as healthy.")
        else:
            st.warning("The model detected a possible plant disease.")

        st.caption(
            "This is an AI prediction, not a confirmed agricultural diagnosis."
        )

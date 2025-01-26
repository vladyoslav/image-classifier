import os
import cv2
import kagglehub
import numpy as np
import streamlit as st
from src import ImageClassifier
from src.utils import load_images_from_categories


# Using @st.cache_resource to cache the classifier
@st.cache_resource
def get_classifier() -> ImageClassifier:
    """
    Function to initialize and train the classifier.
    """
    # Load dataset
    path = os.path.join(
        kagglehub.dataset_download("rahmasleam/intel-image-dataset"),
        "Intel Image Dataset",
    )
    print("Dataset path:", path)

    categories = ImageClassifier.categories
    images, labels = load_images_from_categories(path, categories)

    # Create the classifier
    classifier = ImageClassifier()

    # Train the model
    classifier.fit(images, labels)

    return classifier


def main():
    st.set_page_config(page_title="Image Classifier", page_icon="🖼️")

    st.title("🖼️ Image Classification App")
    st.write(
        "Upload an image, and this app will classify it into one of the predefined categories."
    )

    st.subheader("Choose a file")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    image_placeholder = st.container()
    button_placeholder = st.empty()
    results_placeholder = st.container()

    if uploaded_file is not None:
        # Read the image
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Display the selected image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_placeholder.subheader("Selected image")
        image_placeholder.image(image_rgb, use_container_width=True)

        # Button to classify the image
        button = button_placeholder.button("Classify", type="primary")

        if button:
            with button_placeholder:
                with st.spinner("Processing..."):
                    classifier = get_classifier()

                    # Predict the class
                    predicted_label = classifier.predict(image)

                    # Display the result
                    results_placeholder.subheader("Prediction")
                    results_placeholder.write(
                        f"The image is classified as: **{predicted_label}**"
                    )


if __name__ == "__main__":
    main()

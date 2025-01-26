import os

import cv2
import kagglehub
import streamlit as st

from src import ImageClassifier

def get_image_arrays(dataset_path):
    """
    Retrieve the image arrays and their corresponding labels.

    Args:
        dataset_path (str): Path to the root folder containing images.

    Returns:
        tuple: A tuple containing two lists:
            - List of image arrays (np.ndarray).
            - List of labels for each image.
    """
    categories = list(filter(lambda category: category != ".DS_Store", os.listdir(dataset_path)))

    images = []
    labels = []

    for category in categories:
        category_path = os.path.join(dataset_path, category)
        for img_file in os.listdir(category_path):
            img_path = os.path.join(category_path, img_file)
            if os.path.isfile(img_path) and img_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                image = cv2.imread(img_path)
                if image is not None:  # Ensure the image was successfully loaded
                    images.append(image)
                    labels.append(category)

    return images, labels

@st.cache(allow_output_mutation=True)
def get_classifier():
    path = os.path.join(
        kagglehub.dataset_download("rahmasleam/intel-image-dataset"), "Intel Image Dataset"
    )

    images, labels = get_image_arrays(path)

    # Initialize the classifier
    classifier = ImageClassifier()

    # Train the model on the training set
    classifier.fit(images, labels)

    return classifier


def draw_bar(item):
    key, value = list(item.items())[0]

    col1, col2 = st.columns([12, 1])

    col1.write(key.capitalize())
    col2.write("{:.4f}".format(value))

    st.progress(value)


def main():
    st.set_page_config(page_title="ConvNeXT", page_icon="🐝")
    st.title("🐝 ConvNeXT tiny classifier app")

    st.write("This is an image classification app based on ConvNeXT tiny-sized model.")
    st.write("Model source: https://huggingface.co/facebook/convnext-tiny-224")

    st.subheader("Choose a file")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
    )

    image_placeholder = st.container()
    button_placeholder = st.empty()
    results_placeholder = st.container()

    if uploaded_file is not None:
        image = cv2.imread(uploaded_file)
        image_placeholder.subheader("Selected image")
        image_placeholder.image(image, use_column_width=True)

        button = button_placeholder.button("Classify", type="primary")

        if button:
            with button_placeholder:
                with st.spinner("Processing..."):
                    classifier = get_classifier()

                    results = classifier.predict(image)
                    results_placeholder.subheader("Results")

                    tab1, tab2 = results_placeholder.tabs(["List", "Bar chart"])

                    with tab1:
                        for item in results:
                            draw_bar(item)

                    tab2.bar_chart(results)

    st.caption("by Vladislav Shalnev")


if __name__ == "__main__":
    main()
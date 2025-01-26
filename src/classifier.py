import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List

from src.new_features.feature import HorizontalEdgeCount, CornerCount, VerticalEdgeCount, GreenPixelPercentage, \
    BluePixelPercentage, ContrastMeasure, PerspectiveMeasure, WhitePixelPercentage, SkyPixelRatio, \
    TextureComplexity, ColorDiversity, DominantColor, ShadowPresence, SymmetryMeasure, SharpnessMeasure, \
    AverageBrightness, SaturationVariability


class ImageClassifier:
    """
    A classifier for images using a Random Forest model and a set of predefined features.

    The model predicts probabilities for the following categories:
    ['forest', 'buildings', 'glacier', 'street', 'mountain', 'sea']
    """

    # Class-level feature extractors
    _FEATURE_EXTRACTORS = [
        HorizontalEdgeCount(),
        VerticalEdgeCount(),
        CornerCount(),
        GreenPixelPercentage(),
        BluePixelPercentage(),
        ContrastMeasure(),
        PerspectiveMeasure(),
        WhitePixelPercentage(),
        # HOGFeature(),
        TextureComplexity(),
        SkyPixelRatio(),
        ColorDiversity(),
        DominantColor(),
        ShadowPresence(),
        SymmetryMeasure(),
        SharpnessMeasure(),
        AverageBrightness(),
        SaturationVariability(),
    ]

    _CATEGORIES = ["forest", "buildings", "glacier", "street", "mountain", "sea"]

    def __init__(self, n_estimators: int = 100, random_state: int = 52):
        """
        Initialize the ImageClassifier with a Random Forest model.

        :param n_estimators: Number of trees in the Random Forest.
        :param random_state: Random state for reproducibility.
        """
        self._model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(self._CATEGORIES)

    @classmethod
    def _extract_features(cls, image: np.ndarray) -> np.ndarray:
        """
        Extract features from a single image using the class-level feature extractors.

        :param image: A single image as a numpy array.
        :return: A 1D numpy array of extracted features.
        """
        return np.array(
            [extractor.calculate(image) for extractor in cls._FEATURE_EXTRACTORS]
        )

    def fit(self, images: List[np.ndarray], labels: List[str]) -> None:
        """
        Train the RandomForestClassifier on the provided list of images and labels.

        :param images: List of images as numpy arrays.
        :param labels: List of labels corresponding to each image.
        """
        print("Extracting features from images...")

        # Extract features from all images
        features = [self._extract_features(image) for image in images]

        # Encode the labels
        labels_encoded = self._label_encoder.transform(labels)

        print("Training the Random Forest classifier...")
        self._model.fit(features, labels_encoded)
        print("Training complete!")

    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Predict the probability distribution over categories for a single image.

        :param image: A single image (as a numpy array).
        :return: A dictionary mapping categories to their predicted probabilities.
        """
        print("Extracting features for prediction...")
        features = self._extract_features(image).reshape(1, -1)
        probabilities = self._model.predict_proba(features)[0]
        predicted_probs = {
            category: prob for category, prob in zip(self._CATEGORIES, probabilities)
        }
        return predicted_probs

    def save_model(self, file_path: str) -> None:
        """
        Save the trained RandomForest model to a file.

        :param file_path: Path to save the model.
        """
        print(f"Saving the model to {file_path}...")
        joblib.dump(self._model, file_path)
        print("Model saved successfully!")

    def load_model(self, file_path: str) -> None:
        """
        Load a RandomForest model from a file.

        :param file_path: Path to the model file.
        """
        print(f"Loading the model from {file_path}...")
        self._model = joblib.load(file_path)
        print("Model loaded successfully!")

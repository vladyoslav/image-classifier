import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.features import (
    HorizontalEdgeCount,
    VerticalEdgeCount,
    CornerCount,
    GreenPixelPercentage,
    BluePixelPercentage,
    ContrastMeasure,
    TextureComplexity,
    SkyPixelRatio,
    ShadowPresence,
    SymmetryMeasure,
    SharpnessMeasure,
    AverageBrightness,
)


class ImageClassifier:
    """
    A classifier for images that uses extracted features to predict categories.
    """

    categories = ["forest", "glacier", "street", "sea"]
    features = [
        HorizontalEdgeCount(),
        VerticalEdgeCount(),
        CornerCount(),
        GreenPixelPercentage(),
        BluePixelPercentage(),
        ContrastMeasure(),
        TextureComplexity(),
        SkyPixelRatio(),
        ShadowPresence(),
        SymmetryMeasure(),
        SharpnessMeasure(),
        AverageBrightness(),
    ]

    def __init__(self):
        self.classifier = RandomForestClassifier(random_state=42)

    def _extract_features(self, image: np.ndarray) -> list:
        """
        Extract features from a single image.

        Args:
            image (np.ndarray): The image from which features are extracted.

        Returns:
            list: A list of feature values.
        """
        return [feature.calculate(image) for feature in self.features]

    def fit(self, X_train: list, y_train: list):
        """
        Train the classifier using the provided training data.

        Args:
            X_train (list): A list of images to be used for training.
            y_train (list): The labels corresponding to the images in X_train.
        """
        X_train_features = [self._extract_features(image) for image in X_train]
        self.classifier.fit(X_train_features, y_train)

    def predict(self, image: np.ndarray) -> str:
        """
        Predict the probabilities of categories for a given image.

        Args:
            image (np.ndarray): The image to be classified.

        Returns:
            str: Predicted category.
        """
        features = self._extract_features(image)
        pred = self.classifier.predict([features])[0]
        return pred

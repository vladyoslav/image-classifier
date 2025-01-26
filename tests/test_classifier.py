import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src import ImageClassifier


class MockFeature:
    def calculate(self, image):
        return 1.0


@pytest.fixture
def test_data():
    X_train = [
        np.array([[0, 0], [0, 0]]),
        np.array([[1, 1], [1, 1]]),
        np.array([[0, 1], [1, 0]]),
    ]
    y_train = ["forest", "glacier", "street"]
    return X_train, y_train


def test_feature_extraction():
    classifier = ImageClassifier()
    classifier.features = [MockFeature() for _ in range(12)]
    image = np.array([[0, 0], [0, 0]])
    features = classifier._extract_features(image)

    assert len(features) == 12
    assert all(f == 1.0 for f in features)


def test_fit(test_data):
    X_train, y_train = test_data
    classifier = ImageClassifier()
    classifier.features = [MockFeature() for _ in range(12)]
    classifier.fit(X_train, y_train)

    assert isinstance(classifier.classifier, RandomForestClassifier)
    assert classifier.classifier.n_estimators > 0


def test_predict(test_data):
    X_train, y_train = test_data
    classifier = ImageClassifier()
    classifier.features = [MockFeature() for _ in range(12)]
    classifier.fit(X_train, y_train)

    image = np.array([[0, 0], [0, 0]])
    predicted_label = classifier.predict(image)

    assert predicted_label in classifier.categories


def test_invalid_category_prediction():
    classifier = ImageClassifier()
    classifier.features = [MockFeature() for _ in range(12)]
    X_train = [np.array([[0, 0], [0, 0]])]
    y_train = ["forest"]
    classifier.fit(X_train, y_train)

    invalid_image = np.array([[0, 1], [1, 0]])
    predicted_label = classifier.predict(invalid_image)

    assert predicted_label in classifier.categories

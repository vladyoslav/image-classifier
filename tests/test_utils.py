import pytest
import os
import numpy as np
from unittest.mock import patch
from src.utils import load_images_from_categories


@pytest.fixture
def mock_data():
    categories = ["forest", "glacier", "street"]
    image_paths = {
        "forest": ["image1.jpg", "image2.jpg"],
        "glacier": ["image3.jpg"],
        "street": ["image4.jpg", "image5.jpg"],
    }
    mock_images = {
        "image1.jpg": np.zeros((10, 10, 3), dtype=np.uint8),
        "image2.jpg": np.ones((10, 10, 3), dtype=np.uint8),
        "image3.jpg": np.full((10, 10, 3), 255, dtype=np.uint8),
        "image4.jpg": np.zeros((20, 20, 3), dtype=np.uint8),
        "image5.jpg": np.ones((20, 20, 3), dtype=np.uint8),
    }
    return categories, image_paths, mock_images


@patch("os.listdir")
@patch("cv2.imread")
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
def test_load_images_from_categories(
    mock_path_join, mock_imread, mock_listdir, mock_data
):
    categories, image_paths, mock_images = mock_data
    mock_listdir.side_effect = lambda category_path: image_paths.get(
        os.path.basename(category_path), []
    )

    def mock_read(image_path):
        return mock_images.get(os.path.basename(image_path), None)

    mock_imread.side_effect = mock_read
    path = "/mocked/path/to/dataset"
    images, labels = load_images_from_categories(path, categories)
    assert len(images) == 5
    assert len(labels) == 5
    assert labels[0] == "forest"
    assert images[0].shape == (10, 10, 3)
    assert labels[1] == "forest"
    assert images[1].shape == (10, 10, 3)
    assert labels[2] == "glacier"
    assert images[2].shape == (10, 10, 3)
    assert labels[3] == "street"
    assert images[3].shape == (20, 20, 3)
    assert labels[4] == "street"
    assert images[4].shape == (20, 20, 3)


@patch("os.listdir")
@patch("cv2.imread")
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
def test_load_images_from_categories_no_valid_images(
    mock_path_join, mock_imread, mock_listdir
):
    categories = ["forest", "glacier"]
    image_paths = {
        "forest": ["image1.jpg", "image2.jpg"],
        "glacier": ["image3.jpg"],
    }
    mock_listdir.side_effect = lambda category_path: image_paths.get(
        os.path.basename(category_path), []
    )
    mock_imread.side_effect = lambda image_path: None
    path = "/mocked/path/to/dataset"
    images, labels = load_images_from_categories(path, categories)
    assert len(images) == 0
    assert len(labels) == 0


@patch("os.listdir")
@patch("cv2.imread")
@patch("os.path.join", side_effect=lambda *args: "/".join(args))
def test_load_images_from_categories_missing_category(
    mock_path_join, mock_imread, mock_listdir
):
    categories = ["forest", "nonexistent"]
    image_paths = {
        "forest": ["image1.jpg", "image2.jpg"],
    }
    mock_listdir.side_effect = lambda category_path: image_paths.get(
        os.path.basename(category_path), []
    )

    def mock_read(image_path):
        return np.zeros((10, 10, 3), dtype=np.uint8)

    mock_imread.side_effect = mock_read
    path = "/mocked/path/to/dataset"
    images, labels = load_images_from_categories(path, categories)
    assert len(images) == 2
    assert len(labels) == 2

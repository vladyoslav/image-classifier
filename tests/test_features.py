import cv2
import pytest
import numpy as np
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


@pytest.fixture
def sample_images():
    black_image = np.zeros((100, 100, 3), dtype=np.uint8)
    white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
    gradient_image = np.tile(np.arange(100, dtype=np.uint8), (100, 1))
    gradient_image = cv2.cvtColor(gradient_image, cv2.COLOR_GRAY2BGR)
    green_image = np.zeros((100, 100, 3), dtype=np.uint8)
    green_image[:, :, 1] = 255
    blue_image = np.zeros((100, 100, 3), dtype=np.uint8)
    blue_image[:, :, 2] = 255

    corner_image = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(corner_image, (30, 30), (70, 70), 255, -1)
    corner_image = cv2.cvtColor(corner_image, cv2.COLOR_GRAY2BGR)

    return {
        "black": black_image,
        "white": white_image,
        "gradient": gradient_image,
        "green": green_image,
        "blue": blue_image,
        "corner_image": corner_image,
    }


def test_horizontal_edge_count(sample_images):
    feature = HorizontalEdgeCount()
    result = feature.calculate(sample_images["corner_image"])
    assert result > 0


def test_vertical_edge_count(sample_images):
    feature = VerticalEdgeCount()
    result = feature.calculate(sample_images["gradient"])
    assert result == 0


def test_corner_count(sample_images):
    feature = CornerCount()
    result = feature.calculate(sample_images["corner_image"])
    assert result > 0


def test_green_pixel_percentage(sample_images):
    feature = GreenPixelPercentage()
    result = feature.calculate(sample_images["green"])
    assert result == 1.0


def test_blue_pixel_percentage(sample_images):
    feature = BluePixelPercentage()
    result = feature.calculate(sample_images["blue"])
    assert result == 1.0


def test_contrast_measure(sample_images):
    feature = ContrastMeasure()
    result = feature.calculate(sample_images["gradient"])
    assert result > 0


def test_texture_complexity(sample_images):
    feature = TextureComplexity()
    result = feature.calculate(sample_images["corner_image"])
    assert result > 0


def test_sky_pixel_ratio(sample_images):
    feature = SkyPixelRatio()
    result = feature.calculate(sample_images["white"])
    assert result == 0


def test_shadow_presence(sample_images):
    feature = ShadowPresence()
    result = feature.calculate(sample_images["black"])
    assert result == 1.0


def test_symmetry_measure(sample_images):
    feature = SymmetryMeasure()
    symmetric_image = np.zeros((100, 100), dtype=np.uint8)
    symmetric_image[:, :50] = 255
    symmetric_image[:, 50:] = 255
    result = feature.calculate(symmetric_image)
    assert result == 1.0


def test_sharpness_measure(sample_images):
    feature = SharpnessMeasure()
    result = feature.calculate(sample_images["gradient"])
    assert result > 0


def test_average_brightness(sample_images):
    feature = AverageBrightness()
    result_black = feature.calculate(sample_images["black"])
    assert result_black == 0

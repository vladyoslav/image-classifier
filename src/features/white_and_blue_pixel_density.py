import cv2
import numpy as np
from .feature import IFeature


class WhiteAndBluePixelDensity(IFeature):
    """
    A feature for calculating the density of white and blue pixels in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])

        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        combined_mask = cv2.bitwise_or(white_mask, blue_mask)

        white_and_blue_pixels = cv2.countNonZero(combined_mask)

        total_pixels = image.shape[0] * image.shape[1]

        density = white_and_blue_pixels / total_pixels if total_pixels > 0 else 0

        return density
import cv2
import numpy as np

from .feature import IFeature


class GreenPixelDensity(IFeature):
    """
    A feature for calculating the density of green pixels in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the density of green pixels in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The density of green pixels in the image (ratio of green pixels to total pixels).
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        green_pixels = cv2.countNonZero(green_mask)

        total_pixels = image.shape[0] * image.shape[1]

        green_density = green_pixels / total_pixels if total_pixels > 0 else 0

        return green_density

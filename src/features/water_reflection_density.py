import cv2
import numpy as np
from .feature import IFeature


class WaterReflectionDensity(IFeature):
    """
    A feature for calculating the density of reflections on water surfaces in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the reflection density score on the water surface of the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The reflection density score.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to highlight potential water surfaces with reflections
        _, thresholded = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        reflection_density = len(contours) / (image.shape[0] * image.shape[1])

        return reflection_density

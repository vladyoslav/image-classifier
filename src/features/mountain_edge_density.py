import cv2
import numpy as np
from .feature import IFeature


class MountainEdgeDensity(IFeature):
    """
    A feature extractor that detects sharp edges between the sky and rocky terrain in mountain landscapes.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the edge density in the image, focusing on the separation between the sky and rocky terrain.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The edge density score derived from Sobel edge detection.
        """
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator to detect edges in both horizontal and vertical directions
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the magnitude of the gradient
        magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize the result
        edge_density = np.mean(magnitude)

        return edge_density

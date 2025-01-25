import cv2
import numpy as np
from .feature import IFeature


class OrganicShapesAndCurves(IFeature):
    """
    A feature for calculating the level of organic shapes and curves in an image, which are characteristic of forests.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the level of organic shapes and curves in the image based on contour analysis.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The calculated score for organic shapes and curves.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        edges = cv2.Canny(blurred_image, 100, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate the total length of all contours
        total_contour_length = sum(
            cv2.arcLength(contour, closed=True) for contour in contours
        )

        # Normalize by the number of contours to avoid bias from larger images
        normalized_contour_length = (
            total_contour_length / len(contours) if contours else 0
        )

        return normalized_contour_length

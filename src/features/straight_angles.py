import cv2
import numpy as np

from .feature import IFeature


class StraightAngles(IFeature):
    """
    A feature for calculating the density of straight angles and repetitive patterns in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the density of straight angles (rectangles) in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The density of straight angles (rectangles).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Count rectangles based on contours
        rect_count = 0
        for contour in contours:
            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                rect_count += 1

        total_contours = len(contours)
        return rect_count / total_contours if total_contours > 0 else 0
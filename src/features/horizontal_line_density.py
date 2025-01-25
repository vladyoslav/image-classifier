import cv2
import numpy as np
from .feature import IFeature


class HorizontalLineDensity(IFeature):
    """
    A feature for detecting the density of horizontal lines, potentially related to the horizon, in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the density of horizontal lines (potential horizon lines) in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The density of horizontal lines in the image.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray_image, 50, 150)

        # Detect horizontal lines by applying a Hough transform for lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        # If lines are detected, count how many are horizontal (with a slope close to 0 or pi)
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                if np.isclose(theta, 0) or np.isclose(theta, np.pi):
                    horizontal_lines += 1

        total_lines = len(lines) if lines is not None else 0
        horizontal_density = horizontal_lines / total_lines if total_lines > 0 else 0

        return horizontal_density

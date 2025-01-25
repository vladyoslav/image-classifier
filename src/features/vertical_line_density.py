import cv2
import numpy as np
from .feature import IFeature


class VerticalLineDensity(IFeature):
    """
    A feature for calculating the density of vertical lines in an image,
    which are typically associated with buildings and urban infrastructure.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the density of vertical lines in an image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The density of vertical lines (normalized by the total number of lines).
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        vertical_lines_count = 0
        total_lines_count = 0
        if lines is not None:
            total_lines_count = len(lines)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):  # Check if the line is mostly vertical
                    vertical_lines_count += 1

        # Normalize by the total number of lines, return 0 if no lines were found
        density = (
            vertical_lines_count / total_lines_count if total_lines_count > 0 else 0
        )

        return density

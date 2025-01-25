import cv2
import numpy as np
from .feature import IFeature


class BlueAndTurquoiseGradientDensity(IFeature):
    """
    A feature for calculating the density of blue and turquoise colors with smooth gradients in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the density of blue and turquoise colors along with gradient smoothness in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The density score for blue and turquoise gradients.
        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        lower_turquoise = np.array([81, 50, 50])
        upper_turquoise = np.array([89, 255, 255])

        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        turquoise_mask = cv2.inRange(hsv_image, lower_turquoise, upper_turquoise)

        combined_mask = cv2.bitwise_or(blue_mask, turquoise_mask)

        color_density = cv2.countNonZero(combined_mask) / (
            image.shape[0] * image.shape[1]
        )

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        gradient_smoothness = np.mean(gradient_magnitude)

        overall_score = color_density * (1 / (1 + gradient_smoothness))

        return overall_score

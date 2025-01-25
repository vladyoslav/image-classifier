import cv2
import numpy as np
from .feature import IFeature


class TextureSmoothness(IFeature):
    """
    A feature for calculating the smoothness of textures in an image.
    This feature calculates the standard deviation of pixel intensities,
    where a lower value indicates a smoother texture (more homogeneous).
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the smoothness of the texture based on the standard deviation of pixel intensities.

        Parameters
        ----------
        image : np.ndarray
            The input image in BGR color space.

        Returns
        -------
        float
            The standard deviation of pixel intensities, representing the smoothness of the texture.
            A lower value indicates a smoother, more homogeneous texture.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the standard deviation of pixel intensities
        texture_smoothness = np.std(gray_image)

        return texture_smoothness

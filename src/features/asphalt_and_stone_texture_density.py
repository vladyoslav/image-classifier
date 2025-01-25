import cv2
import numpy as np
from .feature import IFeature


class AsphaltAndStoneTextureDensity(IFeature):
    """
    A feature for calculating the density of asphalt and stone textures in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the texture density based on the presence of asphalt and stone-like patterns in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The texture density score based on asphalt and stone-like textures.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        textural_info = np.abs(laplacian)

        # Threshold the textural information to extract high-texture areas (asphalt, stone)
        _, thresholded = cv2.threshold(textural_info, 20, 255, cv2.THRESH_BINARY)

        texture_area = np.count_nonzero(thresholded)
        total_area = image.shape[0] * image.shape[1]
        texture_density = texture_area / total_area

        return texture_density

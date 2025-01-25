import cv2
import numpy as np
from .feature import IFeature


class ShadowAndTextureContrast(IFeature):
    """
    A feature for calculating the level of shadows and texture contrasts in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the contrast score based on shadows and textures in the image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The contrast score normalized by the standard deviation of the Laplacian.
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # Compute the absolute mean of Laplacian
        raw_contrast_score = np.mean(np.abs(laplacian))

        # Normalize by the standard deviation of the Laplacian
        laplacian_std = np.std(laplacian)
        normalized_contrast_score = raw_contrast_score / (laplacian_std if laplacian_std > 0 else 1)

        return normalized_contrast_score

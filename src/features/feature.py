from abc import ABC, abstractmethod
import numpy as np


class IFeature(ABC):
    """
    An abstract class for defining image feature calculation.
    """

    @abstractmethod
    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate a feature value for a given image.

        Parameters
        ----------
        image : np.ndarray
            A 2D or 3D array representing the image (grayscale or color image).

        Returns
        -------
        float
            The calculated feature value that quantifies a specific aspect of the image.
        """
        pass

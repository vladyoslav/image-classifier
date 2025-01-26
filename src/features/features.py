from abc import ABC, abstractmethod
import numpy as np
import cv2


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


# Feature 1: Count of horizontal edges
class HorizontalEdgeCount(IFeature):
    """
    A feature for counting the number of horizontal edges in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the number of horizontal edges in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The total count of horizontal edges in the image.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        edges = cv2.Canny(gray, 50, 150)
        horizontal_edges = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        return np.sum(horizontal_edges > 0)


# Feature 2: Count of vertical edges
class VerticalEdgeCount(IFeature):
    """
    A feature for counting the number of vertical edges in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the number of vertical edges in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The total count of vertical edges in the image.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        edges = cv2.Canny(gray, 50, 150)
        vertical_edges = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        return np.sum(vertical_edges > 0)


# Feature 3: Count of corners (using Harris Corner Detection)
class CornerCount(IFeature):
    """
    A feature for counting the number of corners in an image using Harris Corner Detection.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the number of corners in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The total count of corners in the image.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        gray = np.float32(gray)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        return np.sum(corners > 0.01 * corners.max())


# Feature 4: Percentage of green pixels (forest indicator)
class GreenPixelPercentage(IFeature):
    """
    A feature for calculating the percentage of green pixels in an image, indicative of forest areas.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the percentage of green pixels in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The percentage of green pixels in the image.
        """
        if len(image.shape) != 3:
            return 0
        green_mask = (image[:, :, 1] > image[:, :, 0]) & (
            image[:, :, 1] > image[:, :, 2]
        )
        return np.sum(green_mask) / (image.shape[0] * image.shape[1])


# Feature 5: Percentage of blue pixels (sea indicator)
class BluePixelPercentage(IFeature):
    """
    A feature for calculating the percentage of blue pixels in an image, indicative of sea areas.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the percentage of blue pixels in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The percentage of blue pixels in the image.
        """
        if len(image.shape) != 3:
            return 0
        blue_mask = (image[:, :, 2] > image[:, :, 1]) & (
            image[:, :, 2] > image[:, :, 0]
        )
        return np.sum(blue_mask) / (image.shape[0] * image.shape[1])


# Feature 6: Contrast (sharp transitions, e.g., mountains against sky)
class ContrastMeasure(IFeature):
    """
    A feature for measuring the contrast in an image, indicative of sharp transitions like mountains against the sky.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the contrast measure of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The variance of the Laplacian, representing the contrast.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()


# Feature 7: Average texture complexity
class TextureComplexity(IFeature):
    """
    A feature for calculating the average texture complexity in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the average texture complexity of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The mean value of edges detected in the image.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        edges = cv2.Canny(gray, 50, 150)
        return edges.mean()


# Feature 8: Percentage of white pixels (glacier or mountain indicator)
class SkyPixelRatio(IFeature):
    """
    A feature for calculating the ratio of sky pixels in an image, based on HSV color space.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the ratio of sky pixels in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The ratio of sky pixels in the image.
        """
        if len(image.shape) != 3:
            return 0
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sky_mask = (hsv[:, :, 0] > 100) & (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 200)
        return np.sum(sky_mask) / (image.shape[0] * image.shape[1])


# Feature 9: Shadow presence
class ShadowPresence(IFeature):
    """
    A feature for detecting the presence of shadows in an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the proportion of shadow pixels in the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The ratio of shadow pixels in the image.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        return np.sum(gray < 50) / (image.shape[0] * image.shape[1])


# Feature 10: Symmetry measure
class SymmetryMeasure(IFeature):
    """
    A feature for measuring the symmetry of an image by comparing it to its horizontally flipped version.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the symmetry measure of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            A value representing the degree of symmetry, where 1 is perfectly symmetrical.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        flipped = cv2.flip(gray, 1)
        diff = np.abs(gray - flipped)
        return 1 - (np.sum(diff) / np.sum(gray))


# Feature 11: Sharpness
class SharpnessMeasure(IFeature):
    """
    A feature for measuring the sharpness of an image using the variance of the Laplacian.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the sharpness of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The variance of the Laplacian, representing sharpness.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian)


# Feature 12: Average brightness
class AverageBrightness(IFeature):
    """
    A feature for calculating the average brightness of an image.
    """

    def calculate(self, image: np.ndarray) -> float:
        """
        Calculate the average brightness of the input image.

        Parameters
        ----------
        image : np.ndarray
            The input image.

        Returns
        -------
        float
            The mean pixel intensity, representing brightness.
        """
        gray = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        )
        return np.mean(gray)

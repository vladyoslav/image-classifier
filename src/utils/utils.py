import os
import cv2
import numpy as np
from typing import List, Tuple


def load_images_from_categories(
    path: str, categories: List[str]
) -> Tuple[List[np.ndarray], List[str]]:
    images: List[np.ndarray] = []
    labels: List[str] = []

    for category in categories:
        category_path = os.path.join(path, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(category)

    return images, labels

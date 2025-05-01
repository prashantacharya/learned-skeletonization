import cv2
import numpy as np

def iterative_thinning(binary_image):
    """
    Perform iterative thinning on a binary image.
    :param binary_image: Input binary image (numpy array with 0s and 1s).
    :return: Thinned binary image.
    """
    # Ensure binary image is in uint8 format
    binary_image = (binary_image * 255).astype(np.uint8)

    # Apply thinning using OpenCV
    thinned = cv2.ximgproc.thinning(binary_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Convert back to binary (0s and 1s)
    return (thinned > 0).astype(np.uint8)
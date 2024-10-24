import math
from scipy import signal
import cv2 as cv
import numpy as np


def gaussian_kernel_2d(size, sigma):
    """Generates a 2D Gaussian kernel with a given variance."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)  # Normalize the kernel


def gaussian_kernel_1d(size, sigma):
    """Generates a 1D Gaussian kernel with a given variance."""
    ax = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-(ax**2) / (2.0 * sigma**2))
    return kernel / np.sum(kernel)


def get_kernel_size(sigma):
    """Computes the kernel size."""
    # 2 * π * σ for the kernel size
    kernel_size = int(round(2 * math.pi * sigma))
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def conv_gaussian_2d(image, kernel_2d: np.ndarray):
    """Applies 2D Gaussian blur to an image with the given variance."""
    image_float = image.astype(np.float32)

    channels = cv.split(image_float)
    convolved_channels = [
        signal.convolve2d(channel, kernel_2d, boundary="symm", mode="same")
        for channel in channels
    ]

    res = cv.merge(convolved_channels)
    res = np.clip(res, 0, 255)
    return np.uint8(res)


def conv_gaussian_separated_1d(image, kernel_1d: np.ndarray):
    """Applies two 1D Gaussian blurs to an image with the given variance."""
    image_float = image.astype(np.float32)

    channels = cv.split(image_float)
    # Apply 1D Gaussian blur in the row direction
    row_blurred_channels = [
        signal.convolve(channel, kernel_1d[:, np.newaxis], mode="same")
        for channel in channels
    ]
    # Apply 1D Gaussian blur in the column direction
    col_blurred_channels = [
        signal.convolve(channel, kernel_1d[np.newaxis, :], mode="same")
        for channel in row_blurred_channels
    ]

    res = cv.merge(col_blurred_channels)
    res = np.clip(res, 0, 255)
    return np.uint8(res)


def validate_blur_equivalence(kernel_2d: np.ndarray, kernel_1d: np.ndarray):
    """
    Validates the equivalence of 2D and 1D Gaussian blurs.

    The separable property of Gaussian kernels allows us to decompose a 2D
    Gaussian kernel into two 1D kernels.
    """

    kernel_1d_combined = kernel_1d[:, np.newaxis] * kernel_1d[np.newaxis, :]
    return np.allclose(kernel_2d, kernel_1d_combined, rtol=1e-5)


def mse(image1, image2):
    err = np.sum((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return float(err)

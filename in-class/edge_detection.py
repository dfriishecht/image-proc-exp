import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import time

from noise import (
    add_gaussian_noise,
    add_poisson_noise,
    add_salt_and_pepper_noise,
)


class Operator(Enum):
    ROBERTS = 1
    PREWITT = 2
    SOBEL = 3
    SOBEL_CUSTOM = 4


def apply_operator(image: np.ndarray, operator: Operator) -> np.ndarray:
    match operator:
        case Operator.ROBERTS:
            kernel_x = np.array([[0, 1], [-1, 0]])
            kernel_y = np.array([[1, 0], [0, -1]])
            x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
            y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
            return cv2.convertScaleAbs(cv2.magnitude(x, y))
        case Operator.PREWITT:
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
            y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
            return cv2.convertScaleAbs(cv2.magnitude(x, y))
        case Operator.SOBEL:
            x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.convertScaleAbs(cv2.magnitude(x, y))
        case Operator.SOBEL_CUSTOM:
            return sobel_operator(image)
        case _:
            raise ValueError("Invalid operator")


def sobel_operator(image: np.ndarray) -> np.ndarray:
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    padded = np.pad(image, 1, mode="constant").astype(np.float32)
    convolved = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = (
                -padded[i, j]
                - 2 * padded[i + 1, j]
                - padded[i + 2, j]
                + padded[i, j + 2]
                + 2 * padded[i + 1, j + 2]
                + padded[i + 2, j + 2]
            )
            y = (
                padded[i, j]
                + 2 * padded[i + 1, j]
                + padded[i + 2, j]
                - padded[i, j + 2]
                - 2 * padded[i + 1, j + 2]
                - padded[i + 2, j + 2]
            )
            # Doing matrix multiplication is slower than the above method since a the kernel has 3 zeros.
            # x = (kernel_x * padded[i : i + 3, j : j + 3]).sum()
            # y = (kernel_y * padded[i : i + 3, j : j + 3]).sum()
            convolved[i, j] = np.hypot(x, y)

    return cv2.convertScaleAbs(convolved)


if __name__ == "__main__":
    image = cv2.imread("assets/IMAGE000.jpg", cv2.IMREAD_COLOR)
    noisy_image_gaussian = add_gaussian_noise(image, 0.1)
    noisy_image_poisson = add_poisson_noise(image)
    noisy_image_salt_pepper = add_salt_and_pepper_noise(image)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gaussian_gray = cv2.cvtColor(noisy_image_gaussian, cv2.COLOR_BGR2GRAY)
    image_poisson_gray = cv2.cvtColor(noisy_image_poisson, cv2.COLOR_BGR2GRAY)
    image_salt_pepper_gray = cv2.cvtColor(
        noisy_image_salt_pepper, cv2.COLOR_BGR2GRAY
    )

    fig, axs = plt.subplots(4, 5, figsize=(25, 15))
    axs[0, 0].imshow(image_gray, cmap="gray")
    axs[0, 0].set_title("Original Image")

    axs[1, 0].imshow(image_gaussian_gray, cmap="gray")
    axs[1, 0].set_title("Gaussian Noise Image")

    axs[2, 0].imshow(image_poisson_gray, cmap="gray")
    axs[2, 0].set_title("Poisson Noise Image")

    axs[3, 0].imshow(image_salt_pepper_gray, cmap="gray")
    axs[3, 0].set_title("Salt and Pepper Noise Image")

    for operator in Operator:
        start_time = time.time()
        result = apply_operator(image_gray, operator)
        elapsed_time = time.time() - start_time
        axs[0, operator.value].imshow(result, cmap="gray")
        axs[0, operator.value].set_title(
            f"{operator.name} ({elapsed_time:.4f}s)"
        )

        start_time = time.time()
        noisy_gaussian_result = apply_operator(image_gaussian_gray, operator)
        elapsed_time = time.time() - start_time
        axs[1, operator.value].imshow(noisy_gaussian_result, cmap="gray")
        axs[1, operator.value].set_title(
            f"{operator.name} ({elapsed_time:.4f}s)"
        )

        start_time = time.time()
        noisy_poisson_result = apply_operator(image_poisson_gray, operator)
        elapsed_time = time.time() - start_time
        axs[2, operator.value].imshow(noisy_poisson_result, cmap="gray")
        axs[2, operator.value].set_title(
            f"{operator.name} ({elapsed_time:.4f}s)"
        )

        start_time = time.time()
        noisy_salt_pepper_result = apply_operator(
            image_salt_pepper_gray, operator
        )
        elapsed_time = time.time() - start_time
        axs[3, operator.value].imshow(noisy_salt_pepper_result, cmap="gray")
        axs[3, operator.value].set_title(
            f"{operator.name} ({elapsed_time:.4f}s)"
        )

    fig.tight_layout()
    plt.show()
    fig.savefig("edge_detection.png")

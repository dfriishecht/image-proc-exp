import numpy as np
import cv2 as cv


def add_gaussian_noise(image: np.ndarray, sigma: float):
    row, col, ch = image.shape
    gaussian = np.random.normal(0, sigma, (row, col, ch))
    noisy_image = np.clip(image + gaussian * 255, 0, 255)
    return noisy_image.astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray, salt_prob=0.001, pepper_prob=0.001
):
    noisy_image = np.copy(image)
    row, col, ch = image.shape

    # Salt noise
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    coords_row = np.random.randint(0, row - 1, num_salt)
    coords_col = np.random.randint(0, col - 1, num_salt)
    noisy_image[coords_row, coords_col, :] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)
    coords_row = np.random.randint(0, row - 1, num_pepper)
    coords_col = np.random.randint(0, col - 1, num_pepper)
    noisy_image[coords_row, coords_col, :] = 0

    return noisy_image


def add_poisson_noise(image: np.ndarray):
    noisy_image = np.clip(np.random.poisson(image), 0, 255)
    return noisy_image.astype(np.uint8)


if __name__ == "__main__":
    img = cv.imread("assets/IMAGE000.jpg", cv.IMREAD_COLOR)

    noisy_image = add_gaussian_noise(img, 0.1)
    cv.imshow("Gaussian Noise", noisy_image)

    noisy_image = add_salt_and_pepper_noise(img)
    cv.imshow("Salt and Pepper Noise", noisy_image)

    noisy_image = add_poisson_noise(img)
    cv.imshow("Poisson Noise", noisy_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

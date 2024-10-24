import numpy as np
import cv2 as cv


def create_random_matrix(rows: int, cols: int) -> np.ndarray:
    """Creates a random matrix with the given number of rows and columns."""
    return np.random.rand(rows, cols)


def sort_matrix(matrix: np.ndarray) -> np.ndarray:
    """Sorts the given matrix in ascending order."""
    return np.sort(matrix, axis=None)


def get_matrix_median(sorted_matrix: np.ndarray) -> float:
    """Returns the median value of the given matrix."""
    if sorted_matrix.size % 2 == 0:
        raise ValueError("Matrix must have an odd number of elements")
    mid = sorted_matrix.size // 2

    return float(sorted_matrix[mid])


def assign_median(matrix: np.ndarray, median: float) -> np.ndarray:
    """Assigns the median value to the center of the given matrix."""
    matrix[matrix.shape[0] // 2, matrix.shape[1] // 2] = median
    return matrix


def median_filter_2d(image: np.ndarray, filter_size: int) -> np.ndarray:
    # Pad the image to handle borders
    pad_size = filter_size // 2
    channels = cv.split(image)

    def median_filter_1d(channel: np.ndarray) -> np.ndarray:
        padded_channel = np.pad(channel, pad_size, mode="reflect")
        filtered_channel = np.zeros_like(channel)
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                # Extract the neighborhood for the current pixel
                neighborhood = padded_channel[
                    i : i + filter_size, j : j + filter_size
                ]
                filtered_channel[i, j] = np.median(neighborhood)
        return filtered_channel

    filtered_image = cv.merge(
        [median_filter_1d(channel) for channel in channels]
    )

    return filtered_image


if __name__ == "__main__":
    img = cv.imread("assets/IMAGE000.jpg", cv.IMREAD_GRAYSCALE)
    matrix = create_random_matrix(5, 5)
    print(matrix)
    sorted_matrix = sort_matrix(matrix)
    print(get_matrix_median(sorted_matrix))
    blurred = median_filter_2d(img, 5)
    cv.imshow("Original Image", img)
    cv.imshow("Filtered Image", blurred)
    cv.waitKey(0)
    cv.destroyAllWindows()

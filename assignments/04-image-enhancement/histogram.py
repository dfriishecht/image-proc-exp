import cv2
import numpy as np
import matplotlib.pyplot as plt


def equalize_histogram(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return gray_image, equalized_image


def plot_histograms(image: np.ndarray, title: str, position: int):
    hist, bins = np.histogram(image.flatten(), 256, (0, 256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.subplot(2, 2, position)
    plt.bar(
        bins[:-1], hist, width=1, color="black", alpha=0.5, label="Histogram"
    )
    plt.plot(cdf_normalized, color="red", label="CDF")
    plt.title(f"Histogram and CDF - {title}")
    plt.xlim([0, 256])
    plt.legend(loc="upper left")


def plot_before_after(
    original: np.ndarray, equalized: np.ndarray, save_as=None
):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(equalized, cmap="gray")
    plt.title("Equalized Image")
    plt.axis("off")

    plot_histograms(original, "Original", 3)
    plot_histograms(equalized, "Equalized", 4)

    plt.tight_layout()
    if save_as:
        plt.savefig(f"results/{save_as}")
    plt.show()


if __name__ == "__main__":
    image_path_1 = "assets/IMAGE000.jpg"
    image_path_2 = "assets/IMAGE002.jpg"
    image_1 = cv2.imread(image_path_1)
    image_2 = cv2.imread(image_path_2)

    original_image_1, equalized_image_1 = equalize_histogram(image_1)
    original_image_2, equalized_image_2 = equalize_histogram(image_2)

    plot_before_after(
        original_image_1,
        equalized_image_1,
        save_as="histogram_equalization_1.png",
    )
    plot_before_after(
        original_image_2,
        equalized_image_2,
        save_as="histogram_equalization_2.png",
    )

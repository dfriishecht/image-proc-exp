import cv2 as cv
from matplotlib import pyplot as plt


def main() -> None:
    img_path = "assets/"
    image_name = "IMAGE000.jpg"
    img = cv.imread(img_path + image_name, cv.IMREAD_GRAYSCALE)

    plt.hist(img.ravel(), 256, (0, 256))
    plt.show()

    threshold = 100
    binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]
    cv.imwrite(f"results/{image_name.split(".")[0]}_binary.png", binary_img)


if __name__ == "__main__":
    main()

import cv2 as cv
from matplotlib import pyplot as plt


def main() -> None:
    img_path = "assets/"
    image_name = "IMAGE000.jpg"
    img = cv.imread(img_path + image_name, cv.IMREAD_GRAYSCALE)

    plt.hist(img.ravel(), 256, (0, 256))
    plt.xlabel("Grayscale value")
    plt.ylabel("Count")
    plt.savefig(
        f'results/{image_name.split(".")[0]}_histogram.png', bbox_inches="tight"
    )
    plt.show()

    threshold = 150
    binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]
    cv.imwrite(f"results/{image_name.split(".")[0]}_binary.png", binary_img)
    cv.imshow("Binary output image", binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

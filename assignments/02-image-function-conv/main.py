# Main working space for assignment 2.
import math
import numpy as np
from scipy import signal
import scipy
from matplotlib import pyplot as plt
import cv2 as cv
import time



def change_brightness(image, amount):
    image = np.clip(image * amount, 0, 255)
    return image

def change_contrast(image, amount):
    bright = (image > 128) * amount
    low = (image <= 128) / amount
    return image * bright + image * low

def change_brightness_alt(image: np.ndarray, amount):
    """Change brightness of alternating pixels"""
    filter = np.ones(image.shape)
    filter[::2][1::2] = amount

    return np.clip(image * filter, 0, 255)

def change_colorspace(image: cv.Mat | np.ndarray, space_in: str, space_out: str):
    match (space_in, space_out):
        case ("BGR", "RGB"):
            return cv.cvtColor(image, cv.COLOR_BGR2RGB)
        case ("RGB", "BGR"):
            return cv.cvtColor(image, cv.COLOR_RGB2BGR)
        case ("BGR", "HSV"):
            return cv.cvtColor(image, cv.COLOR_BGR2HSV)
        case ("HSV", "BGR"):
            return cv.cvtColor(image, cv.COLOR_HSV2BGR)
        case ("RGB", "HSV"):
            return cv.cvtColor(image, cv.COLOR_RGB2HSV)
        case ("HSV", "RGB"):
            return cv.cvtColor(image, cv.COLOR_HSV2RGB)
        case _:
            raise ValueError("Invalid colorspace")

def conv_identity(image: cv.Mat | np.ndarray, kernel_size: int):
    # Create Kernel
    identity = np.zeros([kernel_size, kernel_size])
    identity[math.floor(kernel_size/2), math.floor(kernel_size/2)] = 1

    # Perform convolution
    res = signal.convolve2d(image, identity, boundary='symm')
    return res

def conv_box(image: cv.Mat | np.ndarray, kernel_size):
    # Add code to convolve image with box filter here
     # Create Kernel

    box = np.ones([kernel_size, kernel_size])

    # Perform convolution
    res = signal.convolve2d(image, box, boundary='symm')
    res /= kernel_size**2
    return res

def conv_sharpen(image, kernel_size):
    """Convolve image with sharpen filter"""
    # Create Kernel
    sharpen = np.ones([kernel_size, kernel_size]) * -1
    sharpen[math.floor(kernel_size/2), math.floor(kernel_size/2)] = kernel_size**2
    
    res = signal.convolve2d(image, sharpen, boundary='symm')
    return res

def conv_gaussian(image, kernel_size):
    # Add code to convolve image with gaussian filter here
    #gaussian = np.arrange(30, step = 2).reshape((3,3))
    #gaussian_img = gaussian_filter(image, gaussian, sigma)
    res = scipy.ndimage.gaussian_filter(image, kernel_size)
    return res

def main() -> None:
    img_path = "assets/"
    image_name = "IMAGE002.jpg"
    image = cv.imread(img_path + image_name, cv.IMREAD_GRAYSCALE)
    image_color = cv.imread(img_path+image_name)
    kernel_size = 16
    sigma = 2
    tic = time.time()
    #res = change_colorspace(image_color, "BGR", "HSV")
    res = conv_gaussian(image, 3)
    toc = time.time()
    print(f"Time: {toc-tic}")

    _, ax = plt.subplot_mosaic(
            [["orig", "conv"]], figsize=(15, 10)
        )
    
    ax["orig"].imshow(image, cmap="grey")
    ax["orig"].set_title("Original Image")
    ax["conv"].imshow(res, cmap="grey")
    ax["conv"].set_title("Convolved Image")
    plt.show()


if __name__ == "__main__":
    main()





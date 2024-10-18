import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Function to add Gaussian noise to the image
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gauss * 255  
    noisy_image = np.clip(noisy_image, 0, 255)  
    return noisy_image.astype(np.uint8)

# Function to add salt-and-pepper noise
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image

# median filtering 
def median_filter(image, kernel_size=3):
    return cv.medianBlur(image, kernel_size)

# gaussian filtering 
def gaussian_filter(image, kernel_size=5, sigma=1):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)


img = cv.imread('04-Image-Enhancement/images/IMAGE012.jpg', cv.IMREAD_GRAYSCALE)

# noise to the image
noisy_image_gaussian = add_gaussian_noise(img)
noisy_image_sp = add_salt_pepper_noise(img)

# domain filters
median_filtered = median_filter(noisy_image_sp)
gaussian_filtered = gaussian_filter(noisy_image_gaussian)

# frequency domain low-pass filter
def low_pass_filter(image, radius):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    cv.circle(mask, (ccol, crow), radius, 1, thickness=-1)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back

# applying frequency domain low-pass filter
frequency_filtered_gaussian = low_pass_filter(noisy_image_gaussian, radius=50)
frequency_filtered_sp = low_pass_filter(noisy_image_sp, radius=50)

cv.imwrite("04-Image-Enhancement/results-c/noisy_image_gaussian.jpg", np.uint8(noisy_image_gaussian))
cv.imwrite("04-Image-Enhancement/results-c/noisy_image_s&p.jpg", np.uint8(noisy_image_sp))
cv.imwrite("04-Image-Enhancement/results-c/median_image.jpg", np.uint8(median_filtered))
cv.imwrite("04-Image-Enhancement/results-c/gaussian_image.jpg", np.uint8(gaussian_filtered))
cv.imwrite("04-Image-Enhancement/results-c/frequency_filtered_gaussian.jpg", np.uint8(frequency_filtered_gaussian))
cv.imwrite("04-Image-Enhancement/results-c/frequency_filtered_s&p.jpg", np.uint8(frequency_filtered_sp))



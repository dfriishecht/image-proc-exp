import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('04-Image-Enhancement/images/IMAGE002.jpg', cv.IMREAD_GRAYSCALE)


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# low-pass filter (circular mask)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # Center point
radius = 50  

# circular mask
mask = np.zeros((rows, cols), np.uint8)
cv.circle(mask, (ccol, crow), radius, 1, thickness=-1)

fshift_filtered = fshift * mask

# Inverse Fourier Transform 
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)


cv.imwrite("04-Image-Enhancement/results/grayscale_img2.jpg", img)

# Magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
cv.imwrite("04-Image-Enhancement/results/fourier_transform_img2.jpg", np.uint8(magnitude_spectrum))


magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
cv.imwrite("04-Image-Enhancement/results/filtered_fourier_transform_img2.jpg", np.uint8(magnitude_spectrum_filtered))


cv.imwrite("04-Image-Enhancement/results/low_pass_filtered_image2.jpg", np.uint8(img_back))
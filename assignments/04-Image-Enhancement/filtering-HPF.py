import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


img = cv.imread('04-Image-Enhancement/images/IMAGE012.jpg', cv.IMREAD_GRAYSCALE)


f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)


rows, cols = img.shape
crow, ccol = rows // 2, cols // 2  # Center point
radius = 5  


mask = np.ones((rows, cols), np.uint8)
cv.circle(mask, (ccol, crow), radius, 0, thickness=-1)

fshift_filtered = fshift * mask


f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)



magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift_filtered) + 1)
cv.imwrite("04-Image-Enhancement/results/HPF_filtered_fourier_transform.jpg", np.uint8(magnitude_spectrum_filtered))

cv.imwrite("04-Image-Enhancement/results/high_pass_filtered_image.jpg", np.uint8(img_back))
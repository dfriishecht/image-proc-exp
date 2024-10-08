import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Convert to grayscale
img = cv.imread('images/IMAGE012.jpg', cv.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  

#plt.figure(figsize=(6, 6))
#plt.imshow(magnitude_spectrum, cmap='gray')
#plt.title("Fourier Transform (Magnitude Spectrum)")
#plt.axis('off')
#plt.show()

# Save image
cv.imwrite("results/grayscale_img.jpg", img)
cv.imwrite("results/fourier_transform.jpg", np.uint8(magnitude_spectrum))


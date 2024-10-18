import numpy as np
from matplotlib import pyplot as plt
import math
from scipy import signal
import scipy
import cv2


### CODE FOR FFT OF RECTANGLE ###
rect = np.zeros((30,30), dtype=int)

rect_height = 15
rect_width = 6

start_row = (rect.shape[0] - rect_height) // 2
end_row = start_row + rect_height
start_col = (rect.shape[1] - rect_width) // 2
end_col = start_col + rect_width

rect[start_row:end_row, start_col:end_col] = 1

angle = 45
rect = scipy.ndimage.rotate(rect, angle, reshape=False)

plt.imshow(rect, cmap="grey")
plt.show()

rect_fft = (abs(scipy.fft.fft2(rect, s=(50,50))))

plt.imshow(np.log(rect_fft))
plt.colorbar()
plt.show()
### ###

### CODE FOR FFT OF CIRCLE ###
circle = np.zeros((100,100), dtype=int)

radius = 8

center =  (circle.shape[0]//2, circle.shape[1]//2)

for u in range(circle.shape[0]):
    for v in range(circle.shape[1]):
        if np.sqrt((u-center[0])**2 + (v-center[1])**2) <= radius:
            circle[u, v] = 1

plt.imshow(circle, cmap="grey")
plt.show()

circle_fft = abs(scipy.fft.fftshift((scipy.fft.fft2(circle, s=(100,100)))))

plt.imshow(np.log(circle_fft))
plt.colorbar()
plt.show()
### ###

### CODE FOR FFT OF X ###
x = np.zeros((30,30), dtype=int)

x_size = 15

start_row = (x.shape[0] - x_size) // 2
end_row = start_row + x_size

for i in range(x_size):
    x[start_row + i, start_row + i] = 1
    x[start_row + i, end_row - 1 - i] = 1

plt.imshow(x, cmap="grey")
plt.show()

x_fft = abs(scipy.fft.fftshift(scipy.fft.fft2(x, s=(100,100))))

plt.imshow(np.log(x_fft))
plt.colorbar()
plt.show()


fig, ax = plt.subplot_mosaic(
    [
        ['Rectangle', 'Circle', 'X'],

        ['Rectangle FFT', 'Circle FFT', 'X FFT']
    ]
)

ax['Rectangle'].imshow(rect, cmap='grey')
ax['Circle'].imshow(circle, cmap='grey')
ax['X'].imshow(x, cmap='grey')
ax["Rectangle"].set_title("Rectangle")
ax["Circle"].set_title("Circle")
ax["X"].set_title("X")

ax['Rectangle FFT'].imshow(rect_fft)
ax['Circle FFT'].imshow(circle_fft)
ax['X FFT'].imshow(x_fft)
ax["Rectangle FFT"].set_title("Rectangle FFT")
ax["Circle FFT"].set_title("Circle FFT")
ax["X FFT"].set_title("X FFT")

plt.show()
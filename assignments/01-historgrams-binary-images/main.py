import cv2 as cv
from os.path import expanduser
from matplotlib import pyplot as plt

home = expanduser('~')
img_path = f'{home}/Downloads/IMAGE006.jpg' #Change with your path
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
#plt.hist(img.ravel(),256,[0,256]); plt.show()

threshold = 100
binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]
cv.imwrite('binary_img.png', binary_img)
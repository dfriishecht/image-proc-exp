import cv2 as cv
from os.path import expanduser
from matplotlib import pyplot as plt

home = expanduser('~')
img_path = f'{home}/Downloads/IMAGE002.jpg' #Change with your path
img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
plt.hist(img.ravel(),256,[0,256]); plt.show()

threshold = 200
binary_img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY)[1]
cv.imshow('binary_img.png', binary_img)
cv.waitKey(0)
cv.destoryAllWindows()
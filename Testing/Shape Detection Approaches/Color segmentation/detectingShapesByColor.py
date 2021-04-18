import numpy as np
import cv2 as cv
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator
from matplotlib import pyplot as plt



roi = cv.imread('roi.jpeg')
hsv = cv.cvtColor(roi,cv.COLOR_BGR2HSV)
target = cv.imread('photo2.jpeg')
hsvt = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# calculating object histogram
roihist = cv.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
# normalize histogram and apply backprojection
cv.normalize(roihist,roihist,0,255,cv.NORM_MINMAX)
dst = cv.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
# Now convolute with circular disc
disc = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
cv.filter2D(dst,-1,disc,dst)
# threshold and binary AND
ret,thresh = cv.threshold(dst,50,255,0)
thresh = cv.merge((thresh,thresh,thresh))
res = cv.bitwise_and(target,thresh)
res = np.vstack((target,thresh,res))
cv.imshow('this',dataset_iterator.resize(res,5))


# Take each frame
frame = cv.imread('photo2.jpeg')
# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([0, 0, 140]) # 0,0,140 são bons valores
upper_blue = np.array([179, 85, 255]) # 179, 85 , 255 são bons valores+
# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv.bitwise_and(frame, frame, mask=mask)
res = cv.bitwise_not(res)

cv.imshow('frame', dataset_iterator.resize(frame,10))
cv.imshow('mask', dataset_iterator.resize(mask,10))
cv.imshow('res', dataset_iterator.resize(res,10))

img = cv.imread('photo2.jpeg')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )
plt.imshow(hist,interpolation = 'nearest')
plt.show()


ch = cv.waitKey(0)
if ch == 27:
    pass

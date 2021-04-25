import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("/home/alegria/VCI/02/lego-rot180-6a.jpg")
imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgblur = cv.GaussianBlur(imgray,(5,5),0)
img_canny = cv.Canny(imgblur,threshold1 = 0,threshold2 = 20)

imT = cv.resize(img_canny,(540,940))
cv.imshow('threshold',imT)
kernel = np.ones((2,2),np.uint8)

opening = cv.morphologyEx(img_canny, cv.MORPH_CLOSE, kernel)
ret , thresh = cv.threshold(opening,127,255,0)

imS = cv.resize(opening,(540,940))
cv.imshow('opening',imS)

imgthres = cv.adaptiveThreshold(imgblur,255,cv.ADAPTIVE_THRESH_MEAN_C,\
cv.THRESH_BINARY_INV,101,2)

imS = cv.resize(opening,(540,940))
cv.imshow('opening',imS)
# kernel = np.ones((10,10),np.uint8)
# imT = cv.resize(imgthres,(540,940))
# cv.imshow('threshold',imT)

openingg = cv.morphologyEx(imgthres, cv.MORPH_OPEN, kernel)
ret2 , thresh2 = cv.threshold(openingg,127,255,0)
imTT = cv.resize(imgthres,(540,940))
cv.imshow('threshold',imTT)


contours, hierarchy = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[-2:]
imgdraw = cv.drawContours(img,contours,-1,(0,255),3)


# imS = cv.resize(opening,(540,940))
# cv.imshow('opening',imS)
imC = cv.resize(imgdraw,(540,940))
cv.imshow('contours',imC)
cv.waitKey(0)



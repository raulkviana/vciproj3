import numpy as np
import cv2 as cv
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator
from matplotlib import pyplot as plt

def damkam(x):
    pass
# Take each frame
frame = cv.imread('im2.jpg')

# COLOR EQUALIZATION
img_yuv = cv.cvtColor(frame, cv.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:,:,0] = cv.equalizeHist(img_yuv[:,:,0])

# convert the YUV image back to RGB format
img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

cv.namedWindow('windowNameConfig')
cv.createTrackbar('Hupper','windowNameConfig',0,179,damkam)
cv.createTrackbar('Supper','windowNameConfig',0,255,damkam)
cv.createTrackbar('Vupper','windowNameConfig',0,255,damkam)
cv.createTrackbar('Hlower','windowNameConfig',0,179,damkam)
cv.createTrackbar('Slower','windowNameConfig',0,255,damkam)
cv.createTrackbar('Vlower','windowNameConfig',0,255,damkam)

while (1):

    h = cv.getTrackbarPos('Hupper', 'windowNameConfig')
    s = cv.getTrackbarPos('Supper', 'windowNameConfig')
    v = cv.getTrackbarPos('Vupper', 'windowNameConfig')
    hl = cv.getTrackbarPos('Hlower', 'windowNameConfig')
    sl = cv.getTrackbarPos('Slower', 'windowNameConfig')
    vl = cv.getTrackbarPos('Vlower', 'windowNameConfig')

    # define range of blue color in HSV
    lower = np.array([hl,sl,vl]) # 0,155,0 EXELENTE com tracbar     #    0, 75, 2 bom para  lego       # 0,0,140 são bons valores, para detetar o background
    upper = np.array([h, s, v]) # 179, 255, 255 EXELENTE com tracbar  #      179, 255, 255 bom paralego    #179, 85 , 255 são bons valores, para detetar o background
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', dataset_iterator.resize(frame,10))
    cv.imshow('mask', dataset_iterator.resize(mask,10))
    cv.imshow('res', dataset_iterator.resize(res,10))


    ch = cv.waitKey(1)
    if ch == 27:
        break


# # opening
bw = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5,5)), iterations = 1)
cv.imshow('Open', dataset_iterator.resize(bw, 10))

#Close
bw = cv.dilate(bw, np.ones((9,9)), iterations = 8)
cv.imshow('Close', dataset_iterator.resize(bw, 10))

# Contours
area2=img_output.shape[0]*img_output.shape[0]
contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for cont in contours:
        epsilon = 0.1 * cv.arcLength(cont, True) # 0.025 funcionou bem; 0.01 tambem; 0.005 tambem;
        # Approximations
        approx = cv.approxPolyDP(cont, epsilon, True)
        area = cv.contourArea(approx)
        if area2/(100) <area < area2/12:
            x, y, w, h = cv.boundingRect(cont)


            print(x, y, w, h, end='\n', sep=' ')
            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img_output, [box], 0, (0, 255, 0), 10)

cv.imshow('Final', dataset_iterator.resize(img_output, 10))

if cv.waitKey(0) == 27:
    pass
import numpy as np
import cv2 as cv
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator
from matplotlib import pyplot as plt

def preprocessing(img):
    # COLOR EQUALIZATION
    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    cv.imshow('Equalized', img_output)

    # BLURRING
    blur = cv.GaussianBlur(img_output,(5,5),0)

    return blur


    pass


dataset_iterator.window_with_trackbar(mod_pics_funct=preprocessing, scale_per = 10)
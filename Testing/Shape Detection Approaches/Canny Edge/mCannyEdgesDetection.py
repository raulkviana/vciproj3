import numpy as np
import cv2 as cv
import sys, os
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator
def process_image(img):

    #Grayscaling
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Equalizing
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
    equ = clahe.apply(img2)
    cv.imshow('Equalizing',equ)

    #Blur
    blur = cv.GaussianBlur(equ,(3, 3), 0)
    cv.imshow('Blur',blur)

    #Canny edge
    edges = cv.Canny(blur,10,30)
    cv.imshow('edges',edges)

    #Foreground:
    backSub = cv.createBackgroundSubtractorKNN()
    fg = backSub.apply(img)
    cv.imshow('Foreground',fg)

    return fg

    # #Thresholding
    # #ret,bw = cv.threshold(edges,250,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # bw = cv.adaptiveThreshold(edges,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,31,2)
    #
    # # Close: para obter as formas como deve ser
    # bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((15,15)), iterations = 9)
    #
    # #Erode
    # bw = cv.erode(bw,np.ones((15,15),np.uint8),iterations = 1)
    #
    # # Contours
    # contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # for cont in contours:
    #     epsilon = 0.1 * cv.arcLength(cont, True)
    #     # Two Approximations
    #     approx = cv.approxPolyDP(cont, epsilon, True)
    #
    #     if len(approx) == 4:
    #         x, y, w, h = cv.boundingRect(cont)
    #
    #         #Eliminate small errors
    #         if w>50 and h>50:
    #             print(x, y, w, h, end='\n', sep=' ')
    #             cv.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),5)
    #
    #     else:
    #         cv.drawContours(img3, cont, -1, (255, 0, 0), 1)

dataset_iterator.window_with_trackbar('Window',mod_pics_funct = process_image, scale_per = 10)

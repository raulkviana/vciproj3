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
    blur = cv.GaussianBlur(equ,(7, 7), 0)
    cv.imshow('Blur',blur)

    #Canny edge
    edges = cv.Canny(blur,10,30)
    cv.imshow('edges',edges)

    #Thresholding
    #ret,bw = cv.threshold(edges,250,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # bw = cv.adaptiveThreshold(edges,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,31,2)
    # cv.imshow('Thresholding',bw)

    # Open: eliminar o ruido
    bw = cv.morphologyEx(edges, cv.MORPH_OPEN, np.ones((1,2)), iterations = 1)
    cv.imshow('Opening',bw)

    # Open: eliminar o ruido
    bw = cv.dilate(bw, np.ones((3, 3), np.uint8), iterations=1)
    cv.imshow('Dilate',bw)

    # Close: para obter as formas como deve ser
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((3,3)), iterations = 1)
    cv.imshow('Close',bw)


    # Contours
    contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        epsilon = 0.1 * cv.arcLength(cont, True)
        # Two Approximations
        approx = cv.approxPolyDP(cont, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(cont)

            #Eliminate small errors
            if w>50 and h>50:
                print(x, y, w, h, end='\n', sep=' ')
                cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
        else:
            cv.drawContours(img, cont, -1, (255, 0, 0), 1)

    return img

dataset_iterator.window_with_trackbar('Window',mod_pics_funct = process_image, scale_per = 10)

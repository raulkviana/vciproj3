import numpy as np
import cv2 as cv
import sys, os
from matplotlib import pyplot as plt
import glob

sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator
def process_image(img):

    pass
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

directory_path = '../dataset_iterator/dataset/'

backSub = cv.createBackgroundSubtractorMOG2() # Este metodo resultou em melhores resultados

print('Starting...',end='\n')
count = 0
for pic in glob.glob(directory_path + '*.jpg'):
    img = cv.imread(pic)
    imgRe = dataset_iterator.resize(img,10)
    #Blur
    blur = cv.GaussianBlur(imgRe,(9, 9), 0)
    cannied = cv.Canny(blur,10,30)
    # Foreground:
    fg = backSub.apply(cannied, learningRate=10) # Learning rate esta boa em 10 para MOG2
    #Thresholding
    ret,bw = cv.threshold(fg,250,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # Dilate
    bw = cv.dilate(bw, np.ones((3,3)), iterations=1)
    # # opening
    # bw = cv.morphologyEx(bw, cv.MORPH_OPEN, np.ones((3,3)), iterations = 1)
    # # Closing
    bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((3, 3)), iterations=1)
    #cv.imshow('Foreground'+str(count), bw)

    # Contours
    contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        epsilon = 0.005 * cv.arcLength(cont, True) # 0.025 funcionou bem; 0.01 tambem; 0.005 tambem;
        # Approximations
        approx = cv.approxPolyDP(cont, epsilon, True)
        area = cv.contourArea(approx)
        if 403*309/(4*30) <area < 403*309/17:
            x, y, w, h = cv.boundingRect(cont)


            print(x, y, w, h, end='\n', sep=' ')
            rect = cv.minAreaRect(cont)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(imgRe, [box], 0, (0, 255, 0), 2)


    cv.imshow('Some' + str(count), imgRe)
    count += 1

print('Finished...',end='\n')

cv.imshow('Original',imgRe)
if cv.waitKey(0) == ord('q'):
    cv.destroyAllWindows()
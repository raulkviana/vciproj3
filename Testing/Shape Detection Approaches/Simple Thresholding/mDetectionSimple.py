import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def changeImage(x):
    pass



# Read Image
filename = '4.jpg'
img = cv.imread(filename)
imgRe = cv.resize(img, (960, 540)) # Resize image

# Convert to gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Original Gray ',gray2)

# Thresholding: to find edges
gray = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,21,2)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Thresholded ',gray2)

#Open: Filter image
gray = cv.morphologyEx(gray,cv.MORPH_OPEN,np.ones((6,6),np.uint8), iterations = 1)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Filtered/opened Gray ',gray2)

# Blurring: Filtrar o ruido
#gray = cv.blur(gray,(20,20))
gray = cv.medianBlur(gray,9)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Blur Gray ',gray2)

# Dilate: Para preencher as formas
gray = cv.dilate(gray, np.ones((5,5)) , iterations = 8)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Dilate Gray ',gray2)

# Close: para obter as formas como deve ser
gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, np.ones((15,15)), iterations = 12)
gray2 = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Close Gray ',gray2)

# Contours
contours, hierarchy = cv.findContours(gray2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

for cont in contours:
    epsilon = 0.1 * cv.arcLength(cont, True)
    # Two Approximations
    approx = cv.approxPolyDP(cont, epsilon, True)
    approx = cv.approxPolyDP(approx, epsilon, True)

    if len(approx) == 4:
        x, y, w, h = cv.boundingRect(cont)
        print(x, y, w, h, end='\n', sep=' ')

        #Eliminate small errors
        if w>100 and h>100:
            cv.rectangle(imgRe,(x,y),(x+w,y+h),(0,255,0),5)

    else:
        cv.drawContours(imgRe, cont, -1, (255, 0, 0), 1)

#cv.drawContours(imgRe, contours, -1, (255, 0, 0), 1)
cv.imshow('Contoured Gray ',imgRe)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
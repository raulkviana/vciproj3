import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# Read Image
filename = '4.jpg'
img = cv.imread(filename)
imgRe = cv.resize(img, (960, 540)) # Resize image
cv.imshow('Original Image ',imgRe)

#Turn to gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgRe = cv.resize(gray, (960, 540)) # Resize image
cv.imshow('Gray Image ',imgRe)

# Blur
blur = cv.medianBlur(gray,21)
dst2 = cv.resize(blur, (960, 540)) # Resize image
cv.imshow('Blur',dst2)

# Harris corner detection
gray = np.float32(blur)
dst = cv.cornerHarris(gray,2,3,0.04)
dst2 = cv.resize(dst, (960, 540)) # Resize image
cv.imshow('CornerHarris Image ',dst2)

# Threshold
#dst = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
#print(cv.split(dst),end="\n")
#ret3,dst = cv.threshold(np.float64(dst),0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#dst2 = cv.resize(dst, (960, 540)) # Resize image
#cv.imshow('Threshold Image ',dst2)

#result is dilated for marking the corners, not important
#dst = cv.dilate(dst,None,iterations = 3)
#dst2 = cv.resize(dst, (960, 540)) # Resize image
#cv.imshow('Dilate',dst2)



#dst  = cv.morphologyEx(dst, cv.MORPH_OPEN, np.ones((3,3),np.uint8), iterations = 1)
#dst2 = cv.resize(dst, (960, 540)) # Resize image
#cv.imshow('opening',dst2)

dst = cv.morphologyEx(dst, cv.MORPH_CLOSE, np.ones((6,6),np.uint8), iterations = 40)
dst2 = cv.resize(dst, (960, 540)) # Resize image
cv.imshow('closing',dst2)

# Threshold for an optimal value, it may vary depending on the image.
print(dst.dtype)
val = 0.0001
img[dst>val*dst.max()]=[0,0,255] # Alterar o valor 'val' para ver resultados diferentes

dst2 = cv.resize(img, (960, 540)) # Resize image
cv.imshow('corners ',dst2)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
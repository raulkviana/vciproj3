import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator


# Python3 program change RGB Color
# Model to HSV Color Model
def rgb_to_hsv(r, g, b):
    # R, G, B values are divided by 255
    # to change the range from 0..255 to 0..1:
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # h, s, v = hue, saturation, value
    cmax = max(r, g, b)  # maximum of r, g, b
    cmin = min(r, g, b)  # minimum of r, g, b
    diff = cmax - cmin  # diff of cmax and cmin.

    # if cmax and cmax are equal then h = 0
    if cmax == cmin:
        h = 0

    # if cmax equal r then compute h
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360

    # if cmax equal g then compute h
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360

    # if cmax equal b then compute h
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360

    # if cmax equal zero
    if cmax == 0:
        s = 0
    else:
        s = (diff / cmax) * 255

    # compute v
    v = cmax * 255
    print(h/2,s,v,type(h))
    return np.array([h/2, s, v])


img = cv.imread('im4.jpg')

# Preprocessing pipeline

# resize image
imgRe = dataset_iterator.resize(img,10)

# Grayscaling
gray = cv.cvtColor(imgRe,cv.COLOR_BGR2GRAY)

# Equalize
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=1, tileGridSize=(8,8))
equ = clahe.apply(gray)
#equ = cv.equalizeHist(gray)
cv.imshow('Equalized',equ)

# blur
blur = cv.GaussianBlur(equ,(7,7),0)

#ret, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,5,2) # o penultimo parametro
                                                                                                # igual 5 ficou fixe
cv.imshow('Thresholded',thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)


opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,np.ones((2,3),np.uint8), iterations = 1)
cv.imshow('opening',opening)

thresh=cv.dilate(opening,kernel,iterations=1)
cv.imshow('Dilate',thresh)

closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel, iterations = 10)
cv.imshow('Close/Open',closing)

# sure background area
sure_bg = cv.dilate(closing,kernel,iterations=1)
cv.imshow('Background',sure_bg)

# Finding sure foreground area
dist_transform = cv.distanceTransform(closing,cv.DIST_L2,5) # Calcula a distância ao border mais perto
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0) # Garante-se aqui que o que está presnete na imagem é foreground, visto
                                                                           # que fizemos um distanceTransform para ter uma noção das distâncias a
                                                                           # borda e de seguida usamos essa nova imagem com diferenças de intensidades
                                                                           #, diretamente relacionadas o quão perto se esta da borda, para remover a
                                                                           # parte que se garante que é do forground
cv.imshow('Foreground',sure_fg)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
cv.imshow('Subtraction',unknown) # Border em que não se tem a certeza se é mesmo ou não

# Se objetivo fossse, simplesmente, segmentar o Foreground, bastaria usar erosion, que ajuda a separar objetos numa imageman
# que estejam muito perto um do outro.

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0 # Por a border a 0, de forma a que o algoritmo a entenda como um area em que não se tem a certeza/desconhecida

print(markers)

markers = cv.watershed(imgRe,markers)

print(markers[positions])

#print(imgRe[markers>1])
# print(imgRe)
imgRe2 = imgRe.copy()
positions = np.add(markers > 1,  markers == -1) # posicaos das peças completas, incluindo as bordas
print(markers[positions])
imgRe2[markers > 1] = [0,255,0] # As bordas são representadas por -1 e o foregorund é maior que 1, e o backgorund é 1
imgReNew = imgRe.copy()[markers > 1]

cv.imshow("Random", imgRe)
cv.imshow("Final", imgRe2)

# Get max color
maxColor = np.array([imgReNew.max(0)[0], imgReNew.max(0)[1], imgReNew.max(0)[2]]) # In RGB
# print("\n\n\n\n\nMax BGR:"+str(maxColor), end='\n\n\n\n', )
# maxColorHSV = cv.cvtColor(np.uint16([[maxColor]]),cv.COLOR_BGR2HSV)
maxColorHSV = rgb_to_hsv(imgReNew.max(0)[2], imgReNew.max(0)[1], imgReNew.max(0)[0])
floatVecv = np.vectorize(int)
print("Max HSV:"+str(maxColorHSV), end='\n\n\n\n')

# Get min color
minColor = np.array([imgReNew.min(0)[0], imgReNew.min(0)[1], imgReNew.min(0)[2]])  # In RGB
# print("Min (BGR):"+str(minColor), end='\n\n\n\n')
# minColorHSV = cv.cvtColor(np.uint16([[minColor]]),cv.COLOR_BGR2HSV)
minColorHSV = rgb_to_hsv(imgReNew.min(0)[2], imgReNew.min(0)[1], imgReNew.min(0)[0])
print("Min (HSV):"+str(minColorHSV), end='\n\n\n\n')


matrix1 = np.array([minColorHSV,maxColorHSV])
print(matrix1)
print(matrix1.min(0)[0])
lower = np.array([matrix1.min(0)[0], matrix1.min(0)[1], matrix1.min(0)[2]])
higher = np.array([matrix1.max(0)[0], matrix1.max(0)[1], matrix1.max(0)[2]])
print("Min (HSV):"+str(floatVecv(lower)), end='\n\n\n\n')
print("Max HSV:"+str(floatVecv(higher)), end='\n\n\n\n')

# Get image
imgRe = cv.cvtColor(imgRe, cv.COLOR_BGR2HSV)
last = cv.inRange(imgRe,lower,higher)
cv.imshow("inRange", last)

k = cv.waitKey(0) # Wait for a key/ esperar por um tecla
if k == ord("g"):
    cv.imwrite("savedImage.png", imgRe)
elif k == ord("q"):
    pass



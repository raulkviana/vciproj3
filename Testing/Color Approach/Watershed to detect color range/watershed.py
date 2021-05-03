import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator

img = cv.imread('water_coins.jpg')

# Preprocessing pipeline

# resize image
imgRe = dataset_iterator.resize(img,100)

# Grayscaling
gray = cv.cvtColor(imgRe,cv.COLOR_BGR2GRAY)

# Equalize
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
equ = clahe.apply(gray)

# blur
blur = cv.GaussianBlur(equ,(7,7),0)

ret, thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
cv.imshow('Thresholded',thresh)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 1)
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

markers = cv.watershed(imgRe,markers)
imgRe[markers == -1] = [0,255,0] # As bordas são representadas por -1

cv.imshow("Final", imgRe) # Mostrar imagem
k = cv.waitKey(0) # Wait for a key/ esperar por um tecla
if k == ord("g"):
    cv.imwrite("savedImage.png", imgRe)
elif k == ord("q"):
    print(markers)



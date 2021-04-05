import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('photo2.jpeg',cv.IMREAD_COLOR )
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img3 = img.copy()

#equ = cv.equalizeHist(img2)
#clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
#equ = clahe.apply(img)

#blur = cv.bilateralFilter(equ,9,75,75)
blur = cv.medianBlur(img2,45)

# compute the median of the single channel pixel intensities
#v = np.median(blur)
# apply automatic Canny edge detection using the computed median
#lower = int(max(0, (1.0 - 0.33) * v))
#upper = int(min(255, (1.0 + 0.33) * v))
#edges = cv.Canny(blur, lower, upper)

edges = cv.Canny(blur,10,30)

#ret,bw = cv.threshold(edges,250,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
bw = cv.adaptiveThreshold(edges,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,31,2)

# Close: para obter as formas como deve ser
bw = cv.morphologyEx(bw, cv.MORPH_CLOSE, np.ones((15,15)), iterations = 9)

#Erode
bw = cv.erode(bw,np.ones((15,15),np.uint8),iterations = 1)

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
            cv.rectangle(img3,(x,y),(x+w,y+h),(0,255,0),5)

    else:
        cv.drawContours(img3, cont, -1, (255, 0, 0), 1)


plt.subplot(151),plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(blur,cmap = 'gray')
plt.title('Blured Image'), plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(bw,cmap = 'gray')
plt.title('Threshold edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(155),plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.title('Final Image'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imshow('Final Image', cv.resize(img3,(960,560)))
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()

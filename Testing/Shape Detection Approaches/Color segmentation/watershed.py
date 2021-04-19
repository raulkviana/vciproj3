import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('photo2.jpeg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(thresh, cmap = 'gray')
plt.title(' Mod with threshold'), plt.xticks([]), plt.yticks([])
plt.show()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(opening, cmap = 'gray')
plt.title(' Mod with opening'), plt.xticks([]), plt.yticks([])
plt.show()


# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sure_bg, cmap = 'gray')
plt.title(' Mod with dilate'), plt.xticks([]), plt.yticks([])
plt.show()


# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sure_fg)
plt.title(' Mod with distanceTransform'), plt.xticks([]), plt.yticks([])
plt.show()



# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

plt.subplot(121),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(sure_fg)
plt.title('Finding unknown region'), plt.xticks([]), plt.yticks([])
plt.show()


# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv.imshow("Display window", img) # Mostrar imagem
k = cv.waitKey(10000) # Wait for a key/ esperar por um tecla
if k == ord("g"):
    cv.imwrite("savedImage.png", img)
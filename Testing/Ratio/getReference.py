import cv2 as cv
import dataset_iterator
import numpy as np

REFERENCE_IMAGE_NAME ='imageRef.jpg'
OUTPUT_FILE_NAME = "referenceParameters.npz" # Output file with the with and height

img = cv.imread(REFERENCE_IMAGE_NAME)
img = dataset_iterator.resize(img, 15) # Resize image

# Convert to gray
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('Original Gray ',gray)


# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(gray,(5,5),0)
ret3,th3 = cv.threshold(blur,127,255,cv.THRESH_BINARY_INV)

# Close: para obter as formas como deve ser
bw = cv.morphologyEx(th3, cv.MORPH_CLOSE, np.ones((15,15)), iterations=10)
cv.imshow('BW ',bw)

# Contours
contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
if (len(contours) == 1):
    epsilon = 0.01 * cv.arcLength(contours[0], True)
    # Two Approximations
    approx = cv.approxPolyDP(contours[0], epsilon, True)

    x, y, w, h = cv.boundingRect(approx)

    print(x, y, w, h, end='\n', sep=' ')

    cv.imshow("Image",bw)

    # Print to file
    print("\nSaving in a file")
    from tempfile import TemporaryFile
    outfile = open(OUTPUT_FILE_NAME,'w')
    np.savez(OUTPUT_FILE_NAME, unitSize=int((min(h,w))/2)) # Dividi por 2 porque a imagem é um 2x2

    print("Pixels per unit: ",int((min(h,w))/2))

    print("Success!")

    cv.waitKey(0)

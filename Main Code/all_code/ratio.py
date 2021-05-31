import cv2 as cv
import dataset_iterator
import numpy as np
from numpy import load
import time
# in order to read the color json file
import json

count = 0

# Imports
IMAGE_NAME = "test2.jpg"  # 'imageRef.jpg'testImg
INPUT_FILE_NAME = "referenceParameters.npz"  # Input file with the with an pixels by unit

npz_stuff = load(INPUT_FILE_NAME)
unit_ref = npz_stuff["unitSize"]
print("Pixel per unit:"+str(unit_ref))

'''
#######################################################################################################################
Color detection
#######################################################################################################################
'''

"""
Functions
"""


def resize(img, fx, fy):
    height, width = img.shape[:2]
    size = (int(width * fx), int(height * fy))  # bgr
    img = cv.resize(img, size)

    return img


def imageprocessing(frame, hsv_low, hsv_upper):
    """
    Used to mask the desired region using HSV range
    @param [in] frame : input image
    @param [in] hsv_low : HSV lower range
    @param [in] hsv_upper: HSV upper range

    @param [out] mask_3 : masks the image in order to obtain the filter piece
    @param [out] closing : morphological operation
    @param [out] color_piece : visualize the real part of the target
    """

    # Convert the BGR image to HSV image.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array(hsv_low)
    upper_range = np.array(hsv_upper)

    # Filter the image and get the binary mask, where white represents
    # your target color
    mask = cv.inRange(hsv, lower_range, upper_range)

    # Morphological operation -> closing
    kernel = np.ones((5, 5), np.uint8)
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=8)
    closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=3) # Remove small dots

    # You can also visualize the real part of the target color (Optional)
    color_piece = cv.bitwise_and(frame, frame, mask=closing)

    # Converting the binary mask to 3 channel image, this is just so
    # we can stack it with the others
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    return closing, color_piece, mask_3


def edge_finding(frame, closing, color_piece):
    """
    Edge finding function
    @param [in] frame : input image
    @param [in] closing : morphological operation
    @param [in] color_piece : visualize the real part of the target

    @param [out] edge : edge finding for the original image
    @param [out] edge_color: edge finding in the color piece
    """

    # Find edges
    masked_new = cv.bitwise_and(frame, frame, mask=closing)
    # edge finding for the new image with closing
    gray = cv.cvtColor(masked_new, cv.COLOR_BGR2GRAY)
    # edge finding for the original image
    edge = cv.Canny(gray, 100, 200)
    # edge finding in the color piece
    gray_2 = cv.cvtColor(color_piece, cv.COLOR_BGR2GRAY)
    edge_color = cv.Canny(gray_2, 100, 200)

    return edge, edge_color


def name_contour(closing, frame_contours):
    """
    Contour function
    @param [in] closing : morphological operation
    @param [in] frame_contours : image copy

    @param [out] frame_countours : contours output
    """
    global hsv_color
    contours, hierarchy = cv.findContours(closing,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    minArea = closing.shape[1] * closing.shape[0] / 7000
    maxArea =  closing.shape[1] * closing.shape[0]/4
    print("closing area MAx :",maxArea)
    print("closing area Min :",minArea)

    for c in contours:

        approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
#        print("\napprox: ", approx)
        x, y, w, h = cv.boundingRect(approx)
        print("Area of the piece:",w*h)
        # If the area is not between this range, ignore
        if (minArea < w*h < maxArea):
            aspect_ratio = float(w) / float(h)
            print("aspect:", aspect_ratio)
            print(hsv_color)
            cv.putText(frame_contours, hsv_color, (x-w, y), font, 0.8, (0, 255, 0), 2, cv.LINE_AA)
#            cv.rectangle(frame_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame_contours

'''
#######################################################################################################################
Ratio and corners detection
#######################################################################################################################
'''


def compute_ratio(w, h):
    global unit_ref

    width = int(round(w / (unit_ref-1)))
    height = int(round(h / (unit_ref-1)))

    return (width, height)


def info_about_shape(contour):
    epsilon = 0.01 * cv.arcLength(contour, True)
    # Two Approximations
    approx = cv.approxPolyDP(contour, epsilon, True)

    # Info about the Square
    x, y, w, h = cv.boundingRect(approx)

    return (x, y, w, h)

def findRatio(cont,frame_s):
    x, y, w, h = info_about_shape(cont)

    ratio = compute_ratio(w, h)

    # Write the ratio on image
    cv.putText(frame_s, str(ratio[0]) + "x" + str(ratio[1]), (x + w, y + h), font, 0.5, (0, 0, 0), 1,
               cv.LINE_AA)

def findCorners(cont,frame_s):
    x, y, w, h = info_about_shape(cont)


    #print("point:",point)
    cv.circle(frame_s, (x+w, y), 5, (255, 255, 255), -1)
    cv.circle(frame_s, (x, y+h), 5, (255, 255, 255), -1)
    cv.circle(frame_s, (x, y), 5, (255, 255, 255), -1)
    cv.circle(frame_s, (x+w, y+h), 5, (255, 255, 255), -1)



'''
#######################################################################################################################
MAIN PROGRAM


    
#######################################################################################################################
'''
# Letter font
font = cv.FONT_HERSHEY_SIMPLEX

img = cv.imread(IMAGE_NAME)
img = dataset_iterator.resize(img, 15)  # Resize image

# import the image
frame_s = img.copy()

# #read file
with open('colors.json') as json_file:
    data = json_file.read()

obj = json.loads(data)

for f in obj['colors']:

    hsv_color = (f['color'])
    hsv_low = (f['hsv_low'])
    hsv_upper = (f['hsv_upper'])

    closing, color_piece, mask_3 = imageprocessing(img, hsv_low, hsv_upper)
    #edge,edge_color = edge_finding(frame_s,closing,color_piece)
    frame_contours = name_contour(closing, frame_s)

    # Contours
    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE,  cv.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        epsilon = 0.01 * cv.arcLength(cont, True)
        #  Approximations
        approx = cv.approxPolyDP(cont, epsilon, True)
        #print("cont: ",approx)

        minArea = closing.shape[1] * closing.shape[0] / 7000
        maxArea = closing.shape[1] * closing.shape[0] / 4

        if minArea < cv.contourArea(approx) < maxArea:
            findRatio(approx,frame_s)
            findCorners(approx, frame_s)

cv.imwrite("finalImage2.jpg",frame_s)
cv.imshow("Result", frame_s)

cv.waitKey(0)

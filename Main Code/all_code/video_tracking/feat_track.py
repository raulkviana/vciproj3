import cv2 as cv
import numpy as np
from numpy import load
import time
from lego_tracking import lego_track as Lego
import sys

sys.path.insert(0, r'../')
from Constants import constants_feat
# in order to read the color json file
import json




def apply_mask(frame, hsv_low, hsv_upper):
        """
        Used to mask the desired region using HSV range
        @param [in] frame : input image
        @param [in] hsv_low : HSV lower range
        @param [in] hsv_upper: HSV upper range

        @param [out] closing : morphological operation
        """
        # print("hsv_low: ", hsv_low)
        # print("hsv_upper: ", hsv_upper)

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
        kernel = np.ones((3, 3), np.uint8)
        closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=10)
        closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=3)  # Remove small dots

        return closing

def find_legos(closing, color):
        contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        minArea = closing.shape[1] * closing.shape[0] / 1000
        maxArea = closing.shape[1] * closing.shape[0] / 4

        for c in contours:

            approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
            # print("\napprox: ", approx)
            x, y, w, h = cv.boundingRect(approx)

            # If the area is not between this range, ignore
            if (minArea < w * h < maxArea):
                lego = Lego(color=color)
                lst_legos.append(lego)
                # find contours
                # Find reference point

def update(lego, minDistance):
       # Find new reference point
       # update ratio if necessary
       pass


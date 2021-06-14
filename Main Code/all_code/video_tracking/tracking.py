import cv2 as cv
import numpy as np
from lego import Lego
import sys

#sys.path.insert(0, r'../')
#from Constants import constants_feat
## in order to read the color json file
#import json




def find_legos(closing, color):
    contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    minArea = closing.shape[1] * closing.shape[0] / 1000
    maxArea = closing.shape[1] * closing.shape[0] / 4

    lst_legos = []
    for c in contours:
        approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
        # print("\napprox: ", approx)
        area = cv.contourArea(approx)

        # If the area is not between this range, ignore
        if minArea < area < maxArea:
            ref_point = get_min_y(approx, closing.shape[0])
            print("ref point: ", ref_point)
            lego = Lego(color=color, contour=approx, ref_point=ref_point)
            lst_legos.append(lego)
    return lst_legos


def update(closing, lego, minDistance):

        # Find new reference point
        # update ratio if necessary
    pass


def get_min_y(lst_points, frame_height):
    min_y = frame_height
    for elem in lst_points:
        if elem[0][1] < min_y:
            min_y = elem[0][1]

    return min_y


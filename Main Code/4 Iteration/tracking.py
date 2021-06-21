import cv2 as cv
import numpy as np
from Lego import Lego
from scipy.spatial import distance
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
        approx = cv.approxPolyDP(c, 0.10 * cv.arcLength(c, closed=True), True)
        area = cv.contourArea(approx)

        # If the area is not between this range, ignore
        if minArea < area < maxArea:
            ref_point = get_ref_point(approx, closing.shape[0], closing.shape[1])
            # print("ref point: ", ref_point)
            if len(approx) == 4:
                lego = Lego(color=color, contour=c, ref_point=ref_point, rect_non_rect=True)
            else:
                lego = Lego(color=color, contour=c, ref_point=ref_point, rect_non_rect=False)

            lst_legos.append(lego)

    return lst_legos


def get_key(dict_lego, lego, min_dist):
    for key, values in dict_lego.items():
        if lego.color in values[:][0] and distance.euclidean(lego.ref_point, values[:][1]) <= min_dist:
            return key


def update_dict(dict_lego, lego, min_dist):
    key = get_key(dict_lego, lego, min_dist)

    if key is not None:
        """ update lego's reference point """
        dict_lego[key] = [lego.color, lego.ref_point]

    else:
        """ create new entry in dict """
        dict_lego[len(dict_lego) + 1] = [lego.color, lego.ref_point]


def get_ref_point(lst_points, frame_height, frame_width):
    min_distance = frame_height * frame_width
    for elem in lst_points:
        point = (elem[0][0], elem[0][1])
        dist = distance.euclidean(point, (0, 0))
        if dist < min_distance:
            min_distance = dist
            ref_point = point

    return ref_point


def get_minmax_xy(lst_points, min_max, x_y, frame_width, frame_height):

    if min_max == "min":
        if x_y == "x":
            min_x = frame_width
            for elem in lst_points:
                if elem[0] < min_x:
                    min_x = elem[0]
            return min_x
        # min_y
        else:
            min_y = frame_height
            for elem in lst_points:
                if elem[1] < min_y:
                    min_y = elem[1]
            return min_y
    else:
        # max_x
        if x_y == "x":
            max_x = 0
            for elem in lst_points:
                if elem[0] > max_x:
                    max_x = elem[0]
            return max_x
        # max_y
        else:
            max_y = 0
            for elem in lst_points:
                if elem[1] > max_y:
                    max_y = elem[1]
            return max_y

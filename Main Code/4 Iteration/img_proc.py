import numpy as np
import cv2 as cv


def calc_resize(frame, scale):

    height, width, _ = frame.shape
    new_height = int(height * scale / 100)
    new_width = int(width * scale / 100)
    print("resized width: {}\tresized height: {}".format(new_width, new_height))
    return new_width, new_height


def apply_mask(frame, low_hsv, upper_hsv):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, low_hsv, upper_hsv)
    frame_mask = cv.bitwise_and(frame_hsv, frame_hsv, mask=mask)
    """ convert to gray by using HSV's third channel """
    frame_mask_bw = frame_mask[:,:,2]

    kernel = np.ones((3, 3), np.uint8)
    morph = cv.morphologyEx(frame_mask_bw, cv.MORPH_CLOSE, kernel, iterations=8)
    frame_morph_bw = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel, iterations=5)

    return frame_morph_bw

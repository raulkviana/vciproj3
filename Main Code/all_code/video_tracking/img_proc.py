import numpy as np
import cv2 as cv


def calc_resize(frame, scale):

    height, width, _ = frame.shape
    print("height: {}\twidth: {}".format(height, width))
    new_height = int(height * scale / 100)
    new_width = int(width * scale / 100)
    print("resized height: {}\tresized width: {}".format(new_height, new_width))
    return new_width, new_height


def apply_mask(frame, low_hsv, upper_hsv):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, low_hsv, upper_hsv)
    frame_mask = cv.bitwise_and(frame, frame, mask=mask)
    """ convert to gray by using HSV's third channel """
    frame_mask_bw = frame_mask[:,:,2]

    kernel = np.ones((3, 3), np.uint8)
    morph = cv.morphologyEx(frame_mask_bw, cv.MORPH_CLOSE, kernel, iterations=10)
    frame_morph_bw = cv.morphologyEx(morph, cv.MORPH_OPEN, kernel, iterations=3)

    cv.imshow('frame morph (Gray)', frame_morph_bw)
    """
    DEBUG
    cv.imshow('mask', mask)
    cv.imshow('morph', out_morph)
    cv.imshow('frame', frame)
    """
    return frame_morph_bw


def extract_lego(frame_morph_bw):

    frame_edge = cv.Canny(frame_morph_bw, 100, 200)
    contours, hierarchy = cv.findContours(frame_edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)




def get_contours(frame_bw):
    contours = cv.findContours(cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

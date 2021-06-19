import cv2 as cv
import numpy as np
from numpy import load
import time
from Constants import constants_feat
from lego_tracking import lego_track as Lego
import sys
sys.path.insert(0, r'../')
# in order to read the color json file
import json


class FeatureExtrac:
    def __init__(self, reference=constants_feat.INPUT_FILE_UNIT_REF):
        npz_stuff = load(reference)
        unit_ref = npz_stuff["unitSize"]
        self.unitSize = unit_ref
        self.lst_legos = []
        self.reset_lst = 0
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def get_reference(self, img1, piece_unit_length, outfile_name=constants_feat.INPUT_FILE_UNIT_REF):
        '''
        :param: [in] piece_unit_length: The length of the piece (e.g., if it is a 2x2, its length is 2).
        :param: [in] img_reference:: This image must contain only one rectangle lego piece.

        '''

        # img = self.resize(img,constants_feat.RESIZING_FACTOR, constants_feat.RESIZING_FACTOR)  # Resize image
        img = img1.copy()
        done = False

        while not done:

            # Convert to gray
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # thresholding after Gaussian filtering
            blur = cv.GaussianBlur(gray, (5, 5), 0)
            ret3, th3 = cv.threshold(blur, 127, 255, cv.THRESH_BINARY_INV)

            # Close: para obter as formas como deve ser
            bw = cv.morphologyEx(th3, cv.MORPH_CLOSE, np.ones((10, 10)), iterations=10)
            # cv.imshow('BW ', bw)

            # Contours
            contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if (len(contours)):
                done = True
                epsilon = 0.01 * cv.arcLength(contours[0], True)
                # Two Approximations
                approx = cv.approxPolyDP(contours[0], epsilon, True)

                x, y, w, h = cv.boundingRect(approx)

                self.unitSize = (min(h, w)) / piece_unit_length

                # Print to file
                print("\nSaving in a file")
                from tempfile import TemporaryFile
                outfile = open(outfile_name, 'w')
                np.savez(outfile_name, unitSize=self.unitSize)  # Dividi por 2 porque a imagem é um 2x2

                print("Pixels per unit: ", self.unitSize)

                print("Success!")
            # else:
            #    raise Exception('Reference wasnt obtained')

    def resize(self, img, fx, fy):
        height, width = img.shape[:2]
        size = (int(width * fx), int(height * fy))
        img = cv.resize(img, size)

        return img

    '''
    #######################################################################################################################
    Color detection
    #######################################################################################################################
    '''

    def __imageprocessing(self, frame, hsv_low, hsv_upper):
        """
        Used to mask the desired region using HSV range
        @param [in] frame : input image
        @param [in] hsv_low : HSV lower range
        @param [in] hsv_upper: HSV upper range

        @param [out] mask_3 : masks the image in order to obtain the filter piece
        @param [out] closing : morphological operation
        @param [out] color_piece : visualize the real part of the target
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
        # cv.imshow('Mask',closing)

        # You can also visualize the real part of the target color (Optional)
        color_piece = cv.bitwise_and(frame, frame, mask=closing)

        # Converting the binary mask to 3 channel image, this is just so
        # we can stack it with the others
        mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        return closing, color_piece, mask_3



    def __find_everything(self, closing, frame, color):
        """
        @param [in] closing : morphological operation
        @param [in] frame : image copy
        """

        contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        minArea = closing.shape[1] * closing.shape[0] / 1000
        maxArea = closing.shape[1] * closing.shape[0] / 4

        # Clear list after NUMBER_FRAMES_TO_RESET_LST as passed
        if self.reset_lst >= constants_feat.NUMBER_FRAMES_TO_RESET_LST:
            self.lst_legos.clear()
            self.reset_lst = 0

        self.reset_lst = self.reset_lst + 1

        for c in contours:
            lego = Lego()

            approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
            # print("\napprox: ", approx)
            x, y, w, h = cv.boundingRect(approx)

            # If the area is not between this range, ignore
            if (minArea < w * h < maxArea):
                # aspect_ratio = float(w) / float(h)
                # print("aspect:", aspect_ratio)
                # print(hsv_color)

                lego.color = color
                lego.contour = approx

                if len(approx) == 4:
                    lego.rect = True
                else:
                    lego.rect = False

                # Write to image the color of lego
                #cv.putText(frame, color, (x - w, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2, cv.LINE_AA)

                # Write to image the rect or not of lego
                #if not lego.rect:
                #    cv.putText(frame, 'rect', (x - w, y + round(1.5 * h)), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                #               (235, 206, 135), 2, cv.LINE_AA)
                #else:
                #    cv.putText(frame, 'non-rect', (x - w, y + round(1.5 * h)), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                #               (235, 206, 135), 2, cv.LINE_AA)

                lego.ratio = self.find_ratio(approx, frame)
                #self.find_middle(approx, frame)

                # Add to lego list
                # if not self.__check_in_list(lego):
                self.lst_legos.append(lego)

    '''
    #######################################################################################################################
    Ratio and corners detection
    #######################################################################################################################
    '''

    def __compute_ratio(self, w, h):

        width = round(w / (self.unitSize))
        height = round(h / (self.unitSize))

        return (width, height)

    def __info_about_shape(self, contour):
        '''
        Get info about the bounding box of the shape
        '''

        epsilon = 0.01 * cv.arcLength(contour, True)
        # Approximations
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Info about the Square
        x_y, w_h, angle = cv.minAreaRect(approx)

        return x_y, w_h, angle

    def find_ratio(self, cont, frame_s):
        x_y, w_h, angle = self.__info_about_shape(cont)

        ratio = self.__compute_ratio(w_h[0], w_h[1])
        x_y_lst = list(map(round, list(x_y)))
        w_h_lst = list(map(round, list(w_h)))

        # Write the ratio on image
        #cv.putText(frame_s, str(ratio[0]) + "x" + str(ratio[1]), (x_y_lst[0] + int(w_h_lst[0] / 2), x_y_lst[1]
        #                                                          + int(w_h_lst[1] / 2)), self.font, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        return ratio

    def get_lego_real_size(self, lego):
        ratio_real_world = (lego.ratio[0] * constants_feat.REAL_WORLD_LENGTH_LEGO,
                            lego.ratio[1] * constants_feat.REAL_WORLD_LENGTH_LEGO)

        return ratio_real_world

    def find_middle(self, cont, frame_s):
        x_y, w_h, angle = self.__info_about_shape(cont)
        x_y_lst = list(map(round, list(x_y)))
        cv.circle(frame_s, (int(x_y_lst[0]), int(x_y_lst[1]))
                  , 5, (255, 255, 255), -1)

        # x, y, w, h = info_about_shape(cont)
        # print("point:",point)
        # cv.circle(frame_s, (x + w, y), 5, (255, 255, 255), -1)
        # cv.circle(frame_s, (x, y + h), 5, (255, 255, 255), -1)
        # cv.circle(frame_s, (x, y), 5, (255, 255, 255), -1)
        # cv.circle(frame_s, (x + w, y + h), 5, (255, 255, 255), -1)

    def find_color_ratio(self, img1, json_colors=constants_feat.DEFAULT_COLOR_PATH):
        '''
        Find Color and ratio in an image, with a given json file with colors
        '''

        # img = self.resize(img1, constants_feat.RESIZING_FACTOR,constants_feat.RESIZING_FACTOR)  # Resize image
        img = img1.copy()
        # read file
        with open(json_colors) as json_file:
            data = json_file.read()

        obj = json.loads(data)

        for f in obj['colors']:
            hsv_color = (f['color'])
            hsv_low = (f['hsv_low'])
            hsv_upper = (f['hsv_upper'])

            closing, color_piece, mask_3 = self.__imageprocessing(img, hsv_low, hsv_upper)

            # Find All lego feautures
            self.__find_everything(closing, img1, hsv_color)

    def __check_in_list(self, lego):
        for l in self.lst_legos:
            if l.color == lego.color and l.ratio == lego.ratio and l.rect == lego.rect:
                return True

        return False

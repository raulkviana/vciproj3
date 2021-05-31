import cv2 as cv
import numpy as np
from numpy import load
import time
from Constants import constants_feat
from lego import lego
# in order to read the color json file
import json

class featureExtrac:
    def __init__(self, reference = constants_feat.INPUT_FILE_UNIT_REF):
        npz_stuff = load(reference)
        unit_ref = npz_stuff["unitSize"]
        self.unitSize = unit_ref
        self.lst_legos = []


    def get_reference(self, img_reference, outfile_name, piece_unit_length):
        '''
        :param: [in] piece_unit_length: The length of the piece (e.g., if it is a 2x2, its length is 2).
        :param: [in] img_reference:: This image must contain only one square lego piece.

        '''

        img = cv.imread(img_reference)
        img = resize(img,constants_feat.RESIZING_FACTOR, constants_feat.RESIZING_FACTOR)  # Resize image

        # Convert to gray
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Otsu's thresholding after Gaussian filtering
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        ret3, th3 = cv.threshold(blur, 127, 255, cv.THRESH_BINARY_INV)

        # Close: para obter as formas como deve ser
        bw = cv.morphologyEx(th3, cv.MORPH_CLOSE, np.ones((10, 10)), iterations=10)
        #cv.imshow('BW ', bw)

        # Contours
        contours, hierarchy = cv.findContours(bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if (len(contours) == 1):
            epsilon = 0.01 * cv.arcLength(contours[0], True)
            # Two Approximations
            approx = cv.approxPolyDP(contours[0], epsilon, True)

            x, y, w, h = cv.boundingRect(approx)

            self.unitSize = int((min(h, w)) / piece_unit_length)

            # Print to file
            print("\nSaving in a file")
            from tempfile import TemporaryFile
            outfile = open(outfile_name, 'w')
            np.savez(outfile_name, unitSize=self.unitSize)  # Dividi por 2 porque a imagem é um 2x2

            print("Pixels per unit: ", self.unitSize)

            print("Success!")


    def __resize(self,img, fx, fy):
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
        closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel, iterations=3)  # Remove small dots

        # You can also visualize the real part of the target color (Optional)
        color_piece = cv.bitwise_and(frame, frame, mask=closing)

        # Converting the binary mask to 3 channel image, this is just so
        # we can stack it with the others
        mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

        return closing, color_piece, mask_3

    def __edge_finding(frame, closing, color_piece):
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

    def __find_everything(self,closing, frame,color):
        """
        @param [in] closing : morphological operation
        @param [in] frame : image copy
        """

        contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        minArea = closing.shape[1] * closing.shape[0] / 7000
        maxArea = closing.shape[1] * closing.shape[0] / 4

        for c in contours:
            lego = lego()

            approx = cv.approxPolyDP(c, 0.01 * cv.arcLength(c, True), True)
            #print("\napprox: ", approx)
            x, y, w, h = cv.boundingRect(approx)

            # If the area is not between this range, ignore
            if (minArea < w * h < maxArea):
                #aspect_ratio = float(w) / float(h)
                #print("aspect:", aspect_ratio)
                #print(hsv_color)

                lego.color = color
                lego.contour = approx

                if len(approx) == 4:
                    lego.rect = True
                else:
                    lego.rect = False

                cv.putText(frame, color, (x - w, y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                lego.ratio = findRatio(approx, frame)
                findCorners(approx, frame)

                # Add to lego list
                if not __check_in_list(lego):
                    self.lst_legos.append(lego)

    '''
    #######################################################################################################################
    Ratio and corners detection
    #######################################################################################################################
    '''

    def __compute_ratio(self,w, h):

        width = int(round(w / (self.unitSize - 1)))
        height = int(round(h / (self.unitSize - 1)))

        return (width, height)

    def __info_about_shape(self,contour):
        '''
        Get info about the bounding box of the shape

        '''
        epsilon = 0.01 * cv.arcLength(contour, True)
        # Two Approximations
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Info about the Square
        x, y, w, h = cv.boundingRect(approx)


        return (x, y, w, h)

    def findRatio(self,cont, frame_s):
        x, y, w, h = __info_about_shape(cont)

        ratio = __compute_ratio(w, h)

        # Write the ratio on image
        cv.putText(frame_s, str(ratio[0]) + "x" + str(ratio[1]), (x + w, y + h), self.font, 0.5, (0, 0, 0), 1,
                   cv.LINE_AA)

    def findCorners(self,cont, frame_s):
        for c in cont:
            # print("point:",point)
            cv.circle(frame_s, c, 5, (255, 255, 255), -1)

        #x, y, w, h = info_about_shape(cont)
        # print("point:",point)
        #cv.circle(frame_s, (x + w, y), 5, (255, 255, 255), -1)
        #cv.circle(frame_s, (x, y + h), 5, (255, 255, 255), -1)
        #cv.circle(frame_s, (x, y), 5, (255, 255, 255), -1)
        #cv.circle(frame_s, (x + w, y + h), 5, (255, 255, 255), -1)

    def find_color_ratio(self,image, json_colors = "colors.json"):
        '''
            Find Color and ratio in an image, with a given json file with colors
        '''

        img = cv.imread(image)
        img = dataset_iterator.resize(img, constants_feat.RESIZING_FACTOR)  # Resize image

        # #read file
        with open(json_colors) as json_file:
            data = json_file.read()

        obj = json.loads(data)

        for f in obj['colors']:
            hsv_color = (f['color'])
            hsv_low = (f['hsv_low'])
            hsv_upper = (f['hsv_upper'])

            closing, color_piece, mask_3 = __imageprocessing(img, hsv_low, hsv_upper)

            # Find Color
            find_everything(closing, img,hsv_color)

    def __check_in_list(self, lego):
        for l in self.lst_legos:
            if l == lego:
                return True

        return False




import cv2 as cv
import numpy as np
import tracking
from scipy.spatial import distance

class Lego:
    def __init__(self, color=None, ref_point=None, ratio=None, contour=None, rect_non_rect=None):
        self.color = color
        self.ref_point = ref_point
        self.ratio = ratio
        self.contour = contour
        self.rect = rect_non_rect

    def print_lego(self):
        print("color: {}\nref point: {}" .format(self.color, self.ref_point))

        if self.rect:
            print("Rect")
            if self.contour is not None:
                print("ratio: ", self.ratio)
            if self.ratio is not None:
                print("ratio: ", self.ratio)
        
        elif not self.rect:
            print("Non Rect")

    def __str__(self):

        lego_str = str(self.color) + " , " + str(self.ref_point)
        if self.rect != None:
            if self.rect:
                lego_str += " rect "
            else:
                lego_str += " non_rect "
        if self.ratio != None:
            lego_str += str(self.ratio)

        return lego_str

    def draw_info(self, img, id):
        cv.circle(img, center=self.ref_point, radius=5, color=(255, 255, 255), thickness=3)
        text = "id " + str(id) + " : " + str(self)
        font = cv.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_scale = 0.40
        cv.putText(img, text, self.ref_point, font, color=font_color, fontScale=font_scale, thickness=2)

    def compute_ratio(self, unit_size, frame_width, frame_height):
        rect = cv.minAreaRect(self.contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        min_x = tracking.get_minmax_xy(box, "min", "x", frame_width, frame_height)
        min_y = tracking.get_minmax_xy(box, "min", "y", frame_width, frame_height)
        max_x = tracking.get_minmax_xy(box, "max", "x", frame_width, frame_height)
        max_y = tracking.get_minmax_xy(box, "max", "y", frame_width, frame_height)
        x = distance.euclidean(min_x, max_x)
        y = distance.euclidean(min_y, max_y)
        self.ratio = str(int(x/unit_size)) + "x" + str(int(y/unit_size))

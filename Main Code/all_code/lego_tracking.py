import cv2 as cv
import numpy as np
from lego import lego

class lego_track (lego):
    def __init__(self,ratio = None ,color= None, contour= None, rect_non_rect = None, id = None):
        super().__init__(ratio ,color, contour, rect_non_rect)
        self.id = id

    def get_corners(self):
        '''
        Returns the coordinates of the corners
        '''
        super().get_corners()

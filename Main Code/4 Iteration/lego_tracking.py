import cv2 as cv
import numpy as np
from lego import Lego

class lego_track (Lego):
    def __init__(self,ratio = None ,color= None, contour= None, rect_non_rect = None, id = None):
        super().__init__(ratio ,color, contour, rect_non_rect)
        self.id = id

    def get_corners(self):
        '''
        Returns the coordinates of the corners
        '''
        return super().get_corners()


    def get_center(self):
        '''
        Get center of the lego

        '''
        return super().get_center()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


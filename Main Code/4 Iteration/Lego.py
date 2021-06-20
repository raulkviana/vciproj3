
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


"""
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

"""


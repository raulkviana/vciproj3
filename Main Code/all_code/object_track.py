import math
import cv2 as cv
from lego_tracking import lego_track
from feature_extrac import FeatureExtrac
from lego import Lego

def eucli_dist(o1,o2):
    """
    # 1. Calculate Euclidean Distance of two points    
    :param:
    o1, o2 = two points for calculating Euclidean Distance

    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(o1 + o2)

    return dst


if __name__ == '__main__' :

    cap = cv.VideoCapture("/home/alegria/VCI/OPENCV_COURSE/video_tracking/tracking.mov")


    track = lego_track()
    img1 = cv.imread('/home/alegria/VCI/OPENCV_COURSE/video_tracking/frame164.jpg')
    img2 = cv.imread('/home/alegria/VCI/OPENCV_COURSE/video_tracking/frame220.jpg')
    legos = []
    legos2 = []

    hsv_low = [0,114,127]
    hsv_upper = [28,255,179]

    # creates the object
    im = FeatureExtrac("/home/alegria/VCI/OPENCV_COURSE/vciproj3/Main Code/all_code/reference_param/referenceParameters.npz")

    # x_centroid1,y_centroid1 = im.find_color_ratio(img1,json_colors = "/home/alegria/VCI/OPENCV_COURSE/video_tracking/video_colors.json")
    # legos = im.lst_legos
    # print("lego :\r",legos)
    # print("centroid1:",x_centroid1,y_centroid1)
    # x_centroid2,y_centroid2 = im.find_color_ratio(img2,json_colors = "/home/alegria/VCI/OPENCV_COURSE/video_tracking/video_colors.json")
    # legos2 = im.lst_legos
    # print("Lego2 : \r",legos2)
    # print("centroid2:",x_centroid2,y_centroid2)

# criar dictionario na main
# correr a find_color_ratio a cada frame
# ir ao lst_lego e verificar se esse lego já existe(

    while(1):
        ret, init = cap.read()
        frame = im.resize(init,0.4,0.4)

        x_centroid,y_centroid = im.find_color_ratio(frame,json_colors = "/home/alegria/VCI/OPENCV_COURSE/video_tracking/video_colors.json")
        legos = im.lst_legos
        #print("centroid:",x_centroid,y_centroid)
        print("legos",legos)
        cv.imshow("original frame",frame)




        key = cv.waitKey(1)



        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()

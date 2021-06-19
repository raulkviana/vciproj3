import cv2 as cv
import numpy as np
from scipy.spatial import distance
import tracking
import json

import img_proc as proc


if __name__ == '__main__':

    scale_percent = 100

    cap = cv.VideoCapture("video_tracking_dataset/video_tracking_00.avi")
    ret, frame1 = cap.read()
    dim = proc.calc_resize(frame1, scale_percent)

    # read file
    with open('colors.json') as json_file:
        data = json_file.read()
    # parse file
    obj = json.loads(data)

    while cap.isOpened():

        ret, frame = cap.read()

        cv.imshow('frame main', frame)

        if ret:
            print("--------- NEW FRAME -----------")
            for color in obj['colors']:
                color_name = color['color']
                hsv_low = np.array(color['hsv_low'], np.uint8)
                hsv_upper = np.array(color['hsv_upper'], np.uint8)
                print(" Mask Color: ", color_name)

                frame_resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
                cv.imshow('frame resized', frame_resized)
                frame_bw = proc.apply_mask(frame_resized, hsv_low, hsv_upper)
                cv.imshow('morph', frame_bw)
                lego_lst = tracking.find_legos(frame_bw, color_name)

                print("found {} Legos!" .format(len(lego_lst)))

                #update dict
                #update each lego in dict

        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()

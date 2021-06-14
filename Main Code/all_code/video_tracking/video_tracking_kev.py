import cv2 as cv
import numpy as np
from scipy.spatial import distance
import img_proc as proc
from feature_extrac import FeatureExtrac as FE
from lego_tracking import lego_track

minDistance = 100 # Pixels
id_cnt = 0 # Id counter
FRAMES_TO_SKIP = 100
out_lst = []
DISTANCE_FROM_TOP = 100
'''
       {
            "color": "blue",
            "hsv_low": [105, 132, 173],
            "hsv_upper":[114, 255, 255]
        },
       {
            "color": "green",
            "hsv_low": [60, 167, 127],
            "hsv_upper":[85, 255, 206]
        },
       {
            "color": "orange",
            "hsv_low": [0, 147, 135],
            "hsv_upper":[6, 255, 255]
        }
'''
def calculateDistance(lego1, lego2):
    if lego1 != None and lego2 != None:
        lego_center1 = lego1.get_center()
        lego_center2 = lego2.get_center()

        return distance.euclidean((lego_center1[0],lego_center1[1]),(lego_center2[0],lego_center2[1]))
    else:
        return 100000

def check_lego_in_main_lst(lego_lst1, main_lego_lst):
        if lego_lst1 != []:
            if main_lego_lst != []:
                for l1 in lego_lst1:
                    for l2 in main_lego_lst:
                            dist = calculateDistance(l1, l2)
                            print('Distance between legos',dist)
                            # It's the same lego
                            if dist <= minDistance:
                                l2.contour = l1.contour
                            # It's a new lego
                            else:
                                main_lego_lst.append(l1)

            else:
                main_lego_lst.extend(lego_lst1)

def verify_lego_gone(main_lego_lst, out_lst):

    for l in main_lego_lst:
        coord = l.get_center()

        # Remover lego, caso este esteja quase a sair
        if(distance.euclidean(coord[1],0) < DISTANCE_FROM_TOP):
            if not check_in_list(l,out_lst):
                out_lst.append(l)

def add_id(lego_lst):
    global id_cnt

    for l in lego_lst:
        if l.id == None:
            l.id = id_cnt
            id_cnt = id_cnt + 1

def check_in_list(lego, out_lst):
        for l in out_lst:
            if l.id == lego.id:
                return True

        return False


if __name__ == '__main__':
    fe = FE()
    main_lego_lst = []
    sec_lego_lst = []

    cap = cv.VideoCapture("./video_tracking_dataset/edit_dataset2.mp4")
    ret, frame1 = cap.read()
    frame_control = FRAMES_TO_SKIP

    print('Starting tracking')

    while cap.isOpened():
        ret, frame = cap.read()
        if(frame_control <= 0):

            # if frame is read correctly ret is True
            if ret:
                # Resize image
                frame_resized = fe.resize(frame, 0.15, 0.15)

                # Look for legos
                fe.find_color_ratio(frame_resized)
                sec_lego_lst.extend(fe.lst_legos) # Add it to the secondary lst

                # Check if legos are already in the lst and update there positions
                check_lego_in_main_lst(sec_lego_lst,main_lego_lst)

                # Add id to the legos
                add_id(main_lego_lst)

                verify_lego_gone(main_lego_lst,out_lst)

                sec_lego_lst.clear()
                if cv.waitKey(1) == ord('q'):
                    break

                cv.imshow('Output Window', frame_resized)

            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame_control = FRAMES_TO_SKIP
        else:
            frame_control = frame_control - 1

    cap.release()
    cv.destroyAllWindows()

    print('I found ', len(out_lst), ' legos!')

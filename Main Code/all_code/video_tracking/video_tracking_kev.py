import cv2 as cv
import numpy as np
from scipy.spatial import distance
import img_proc as proc
from feature_extrac_track import FeatureExtrac as FE
from lego_tracking import lego_track


minDistance = 500 # Pixels
PredictionDistance = 10 # Pixels
id_cnt = 0 # Id counter
FRAMES_TO_SKIP = 300
DISTANCE_FROM_TOP = 1000
resize_amount = 0.15

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
'''

def calculateDistance(lego1, lego2):
    if lego1 != None and lego2 != None:
        lego_center1 = lego1.get_center()
        lego_center2 = lego2.get_center()

        return distance.euclidean((lego_center1[0],lego_center1[1]),(lego_center2[0],lego_center2[1]))
    else:
        return 100000

def check_lego_in_main_lst(lego_lst1, main_lego_lst):
        # If no lego found, ignore
        if lego_lst1 != []:
            # If list is empty, fill it
            if main_lego_lst != []:

                for l1 in lego_lst1:

                    for l2 in main_lego_lst:

                        # If lego is still in the picture, check it's position
                        if l2.get_center()[1] > 0:

                            # Calculate distance from lego
                            dist = calculateDistance(l1, l2)

                            # It's the same lego
                            if dist <= minDistance:
                                l2.contour = l1.contour

                                # Predict next position
                                for c in l2.contour:
                                    c[0][1] = c[0][1] - PredictionDistance # next position

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

def count_legos(main_lego_lst, out_lst):
    count = len(main_lego_lst) - 1
    for l in range(0,count,1):
            if l>=len(main_lego_lst)-1:
                break
            # Remover lego caso tenham o mesmo id
            if check_in_list(main_lego_lst[l],out_lst):
                    main_lego_lst.pop(l)
                    count = count - 1


    return len(main_lego_lst) + len(out_lst)

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
    main_lego_lst = [] # List containing legos that are still in the image
    sec_lego_lst = [] # List with legos obtained through feature extraction
    out_lst = [] # List with legos that are out of the image

    cap = cv.VideoCapture("./video_tracking_dataset/edit_dataset2.mp4")
    #ret, frame1 = cap.read()
    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv.VideoWriter_fourcc(*'MJPG')
    out = cv.VideoWriter('output.avi', fourcc, 1, (int(frame_width*resize_amount), int(resize_amount*frame_height)))

    frame_control = FRAMES_TO_SKIP

    print('Starting tracking')

    while cap.isOpened():
        ret, frame = cap.read()
        if(frame_control <= 0):

            # if frame is read correctly ret is True
            if ret:
                # Resize image
                frame_resized = fe.resize(frame, resize_amount, resize_amount)

                # Look for legos
                fe.find_color_ratio(frame_resized)
                sec_lego_lst.extend(fe.lst_legos) # Add it to the secondary lst

                # Check if legos are already in the lst and update there positions
                check_lego_in_main_lst(sec_lego_lst,main_lego_lst)

                # Add id to the legos
                add_id(main_lego_lst)

                # Verify if lego is still in the video
                verify_lego_gone(main_lego_lst,out_lst)

                # Reset secondary list
                sec_lego_lst.clear()

                if cv.waitKey(1) == ord('q'):
                    break

                cv.imshow('Output Window', frame_resized)
            else:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_control = FRAMES_TO_SKIP
            out.write(frame_resized)

        else:
            # Number of frames left to skip
            frame_control = frame_control - 1

    cap.release()
    out.release()

    cv.destroyAllWindows()

    print('Finished!')
    print('Computing number of legos..')
    #print('I found ', count_legos(main_lego_lst, out_lst), ' legos!')
    print('Main lst size: ', len(main_lego_lst))
    print('Out lst size: ', len(out_lst))
    print('Total IDs: ', id_cnt)

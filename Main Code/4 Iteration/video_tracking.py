import cv2 as cv
import numpy as np
import tracking
import json
import img_proc as proc
from Lego import Lego


if __name__ == '__main__':

    scale_percent = 100
    MIN_DISTANCE = 50
    LEGO_UNIT_SIZE = 17

    cap = cv.VideoCapture("video_tracking_dataset/video_tracking_04.avi")
    ret, frame1 = cap.read()
    dim = proc.calc_resize(frame1, scale_percent)
    min_roi = int(dim[1]/3)
    max_roi = int(dim[1]*5/6)

    # read file
    with open('colors_video_tracking.json') as json_file:
        data = json_file.read()
    # parse file
    obj = json.loads(data)

    legos_dict = {}
    while cap.isOpened():
        ret, frame = cap.read()
        lego_lst = []
        if ret:
            #print("--------- NEW FRAME -----------")
            all_masks = np.zeros((dim[1], dim[0]), dtype=np.uint8)
            frame_resized = cv.resize(frame, (dim), interpolation=cv.INTER_AREA)
            cv.imshow('frame resized', frame_resized)

            for color in obj['colors']:
                color_name = color['color']
                hsv_low = np.array(color['hsv_low'], np.uint8)
                hsv_upper = np.array(color['hsv_upper'], np.uint8)
                frame_bw = proc.apply_mask(frame_resized, hsv_low, hsv_upper)
                all_masks = cv.bitwise_or(all_masks, frame_bw)
                lego_color_lst = tracking.find_legos(frame_bw, color_name)
                lego_lst.extend(lego_color_lst)

            cv.imshow('all masks', all_masks)

            for lego in lego_lst:
                if lego.ref_point[1] > 15:
                    tracking.update_dict(legos_dict, lego, MIN_DISTANCE)
                    key = tracking.get_key(legos_dict, lego, MIN_DISTANCE)
                    legos_dict[key].append(lego)

            if legos_dict:
                print("dict content:")
                for key, value in legos_dict.items():
                    lego = value[2]
                    if min_roi < lego.ref_point[1] < max_roi and lego.rect is True:
                        Lego.compute_ratio(lego, LEGO_UNIT_SIZE, dim[0], dim[1])
                    print("{} : {}" .format(key, str(lego)))
                    lego.draw_info(frame_resized, key)

                cv.imshow('draw info', frame_resized)

        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        k = cv.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

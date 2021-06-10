import cv2 as cv
import numpy as np
from scipy.spatial import distance

import img_proc as proc


if __name__ == '__main__':

    scale_percent = 300

    cap = cv.VideoCapture("video_tracking_dataset/video_tracking_00.avi")
    ret, frame1 = cap.read()
    dim = proc.calc_resize(frame1, scale_percent)

    dst = distance.euclidean((0, 0, 0), (1, 1, 1))
    print(dst)

    low_orange = (0, 113, 147)
    upper_orange = (6, 212, 203)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if ret:
            frame_resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            frame_bw = proc.apply_mask(frame_resized, low_orange, upper_orange)



            if cv.waitKey(1) == ord('q'):
                break

        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

    cap.release()
    cv.destroyAllWindows()

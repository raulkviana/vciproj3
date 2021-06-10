"""
tentativa de video editor
"""

import cv2 as cv


if __name__ == '__main__':

    scale_percent = 250

    cap = cv.VideoCapture("video_tracking_dataset/output05.avi")
    ret, frame1 = cap.read()

    height, width, _ = frame1.shape
    dim = (width, height)
    print(height, width)
    """
    new_height = int(height * scale_percent / 100)
    new_width = int(width * scale_percent / 100)
    dim = (new_width, new_height)
    """

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('video_tracking_dataset/video_tracking_05.avi', fourcc, 20.0, dim)

    while cap.isOpened():
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if ret:

            frame_resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
            #frame_resized = frame_resized[:,120:581,:]
            frame_resized[:,0:110,:] = (0,0,0)
            frame_resized[:,580:,:] = (0,0,0)
            out.write(frame_resized)

            cv.imshow('frame', frame_resized)

            if cv.waitKey(1) == ord('q'):
                break

        else:
            print("Can't receive frame (stream end?). Exiting ...")
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Video was successfully saved.")

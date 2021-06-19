import cv2 as cv
import numpy as np
from scipy.spatial import distance
from feature_extrac_track import FeatureExtrac as FE
from lego_tracking import lego_track

backSub = cv.createBackgroundSubtractorMOG2()#cv.createBackgroundSubtractorKNN()# # Este metodo resultou em melhores resultados
cap = cv.VideoCapture("./video_tracking_dataset/edit_dataset2.mp4")
fe = FE()

firstFrame = None
print('Starting tracking')
while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if ret:
        # Resize image
        frame_resized = fe.resize(frame, 0.30, 0.30)
        cv.imshow('Original IMG', frame_resized)

        """
        # GrayScale image
        gray = cv.cvtColor(frame_resized, cv.COLOR_BGR2GRAY)

        # create a CLAHE object (Arguments are optional).
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)

        ret3, th3 = cv.threshold(cl1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        cv.imshow('Otzu\' Threshold', th3)

        ret3, th3 = cv.threshold(cl1, 80, 255, cv.THRESH_BINARY)
        cv.imshow('Threshold', th3)

        th3 = cv.adaptiveThreshold(cl1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                   cv.THRESH_BINARY, 11, 2)
        cv.imshow('Adaptive Threshold', th3)
        """

        try:
            if firstFrame == None:
                firstFrame = frame_resized

        except:
            newFrame = firstFrame - frame_resized
            #newFrame [:,:,2] = 0
            cv.imshow('Subtraction', newFrame)

            median = cv.medianBlur(newFrame, 7)
            cv.imshow('medianBlur', median)
            mask = cv.inRange(median, np.array([10,0,20]), np.array([179,255,230]))


            cv.imshow('Subtraction + Threshold', mask)
            #fg = backSub.apply(frame_resized, learningRate=5)  # Learning rate esta boa em 10 para MOG2
        user_command = cv.waitKey(1)
        if user_command == ord('q'):
            break
        elif user_command == ord('u'):
            firstFrame = frame_resized



    else:
        print("Can't receive frame (stream end?). Exiting ...")
        break

if cv.waitKey(-1) == ord('q'):

    cap.release()
    cv.destroyAllWindows()


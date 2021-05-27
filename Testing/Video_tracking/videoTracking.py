# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2

'''
Função para verificar se o lego se encontra na lista
'''
def findLego(legoList, high, low,lego):
    if high >= low:
            mid = (high + low) // 2
            if legoList[mid].color == lego.color:
                return legoList[mid]

            elif legoList[mid].color > lego.color:
                return binarySearch(legoList, low, mid - 1, lego)
            else:
                return binarySearch(legoList, mid + 1, high, lego)

    else:
            return None
'''
Função para organizar a lista de legos por cor
'''
def sortListByColor(legoList):
    legoList.sort(key= lambda x: x.color)


'''
Lista com diversos legos
'''
legoList = []


OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
trackers = cv2.MultiTracker_create()


vs = cv2.VideoCapture("vid1.mov")
# initialize the FPS throughput estimator
fps = None

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    ret, frame = cap.read()
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=500)
    # check to see if we are currently tracking an object
    # grab the new bounding box coordinates of the object
    (success, box) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame
    '''
    Nesta ciclo for tens que verificar se um dado lego esta ou não ainda no frame.
    Verificar a variação de coordenadas para cada um dos legos e assim verificar se este ainda está no vídeo ou não.
    '''
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track

    '''
    Editar aqui para poderes fazer o track dos os legos!
    '''
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        ''' 
        Nesta linha tens que obter uma bounding box para os diversos legos
        '''
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

        '''
        Add all new legos found to the tracker
        '''
        trackers.add(tracker, frame, box)

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# Release the pointer
vs.release()
# close all windows
cv2.destroyAllWindows()
#!/home/pi/Desktop/venv 
import numpy as np
import cv2 as cv
import glob
# termination criteria
rows = 8
columns = 6
sizeRealWorld = 24
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, sizeRealWorld, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((columns*rows,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# Start camera
cam = cv.VideoCapture(0)
counter = 0
while(True):
    ret, frame = cam.read()
    #cv.imshow("showing",frame)
    cv.waitKey(2)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (rows,columns), None)
 

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Hello!!!")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (rows,columns), corners2, ret)
        cv.imshow('img'+str(counter), frame)
        cv.waitKey(5)
        counter = counter + 1
        
    if counter ==15:
        break

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret,mtx,dist,rvecs,tvecs)
if(cv.waitKey(0) == ord('s')):

    cv.destroyAllWindows()
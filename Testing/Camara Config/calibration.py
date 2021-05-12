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
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (rows,columns), None)
 

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(frame, (rows,columns), corners2, ret)
        cv.imwrite("img"+str(counter)+".png",frame)
        counter = counter + 1

    
    cv.waitKey(1000)
    cv.imshow("showing",frame)
    
    if counter == 25:
        break

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Distortion coeficients: ",str(dist))
print("Rotation vector: ",str(rvecs))
print("Translation vector: ",str(tvecs))
print("Intrinsic parameters: ",str(mtx))

# print to file
print("Saving in a file")
from tempfile import TemporaryFile
outfile = open('camera.npz','w')
np.savez('camera.npz', intrinsics=mtx, distortion=dist, rotation=rvecs, translation=tvecs)


# Re-projection error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )


cv.destroyAllWindows()
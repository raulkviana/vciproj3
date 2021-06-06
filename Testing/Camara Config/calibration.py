#!/home/pi/Desktop/venv 
import numpy as np
import cv2 as cv
import glob

# Chessboard info
rows = 8
columns = 6
sizeRealWorld = 24 # mm

# Constants
NUMBER_OF_PICS = 25 # Number of pictures to be taken
FRAMES_PER_SECOND = 200 # fps
OUTPUT_FILE_NAME = "camara.npz" # Output file with the calibration parameters


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, sizeRealWorld, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((columns*rows,3), np.float32)
objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Start camera
cam = cv.VideoCapture(0)
counter = 0 # Count for the images name


while(True):

    ret, frame = cam.read() # Get frame from camara

    # Transform image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (rows,columns), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        # Adding points found to the vectors
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, (rows,columns), corners2, ret)

        # Save image
        cv.imwrite("img"+str(counter)+".png",frame) 
        counter = counter + 1

    
    cv.waitKey(FRAMES_PER_SECOND) # Delay
    cv.imshow("Showing",frame) # Show image
    
    if counter == NUMBER_OF_PICS:
        break

# Calibration parameters
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Present results 
print("Distortion coeficients: ",str(dist))
print("Rotation vector: ",str(rvecs))
print("Translation vector: ",str(tvecs))
print("Intrinsic parameters: ",str(mtx))

# Print to file
print("\n\n\n\nSaving in a file")
from tempfile import TemporaryFile
outfile = open(OUTPUT_FILE_NAME,'w')
np.savez(OUTPUT_FILE_NAME, intrinsics=mtx, distortion=dist, rotation=rvecs, translation=tvecs)


# Re-projection error
print("\n\n\n\nCalculate Re-projection error")
mean_error = 0 
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist) # Project points obtained before with the parameters calculated
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2) # Calculate the norm between the point obtained in the line above and the
    mean_error = mean_error + error # Add to the variable mean_error, to calculate the mean afterwards

print( "Mean Re-projection error: {}".format(mean_error/len(objpoints)) )

cv.waitKey(0) # Wait for any key to quit
cv.destroyAllWindows()
#!/home/pi/Desktop/venv 
import numpy as np
import cv2 as cv
import glob
from numpy import load

# Load all the images
images = glob.glob('/home/pi/Desktop/calibration/img*.png')

# Get the camera parameters
data = load("camera.npz")
lst = data.files 

count = 0 # Variable for the name of the images

for img in images:

    img2 = cv.imread(img) # Read image
    h, w = img2.shape[:2] # Get height and width
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(data["intrinsics"], data["distortion"], (w,h), 1, (w,h)) # Calculate new camera matrix, optimized
    
    # Undistort
    dst = cv.undistort(img2, data["intrinsics"], data["distortion"], None, newcameramtx)

    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    fileOutName = 'calibresult'+str(count)+'.png'
    cv.imwrite(fileOutName, dst) # Write image

    # Update counter
    count = count + 1
    
    


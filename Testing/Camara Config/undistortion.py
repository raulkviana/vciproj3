#!/home/pi/Desktop/venv 
import numpy as np
import cv2 as cv
import glob
from numpy import load

images = glob.glob('/home/pi/Desktop/calibration/img*.png')

data = load("camera.npz")
lst = data.files
count = 0

for img in images:
    img2 = cv.imread(img)
    h, w = img2.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(data["intrinsics"], data["distortion"], (w,h), 1, (w,h))
    
    # undistort
    dst = cv.undistort(img2, data["intrinsics"], data["distortion"], None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    fileOutName = 'calibresult'+str(count)+'.png'
    cv.imwrite(fileOutName, dst)
    count = count + 1
    
    


import numpy as np
import cv2
import glob

# Board Size
board_w = 7;
board_h = 5;

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def  FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts the image to gray
    print(gray.shape[::-1])
    r, c = cv2.findChessboardCorners(gray, (board_w,board_h), None)

    # If found, display image with corners
    if r == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), c, r)
        cv2.imshow('img',img)
        cv2.waitKey(500)

    return r, c

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((board_w*board_h,3), np.float32)
print(np.mgrid[0:board_w,0:board_h])
print(np.mgrid[0:board_w,0:board_h].T.reshape(-1,2))
objp[:,:2] = np.mgrid[0:board_w,0:board_h].T.reshape(-1,2)

cv2.waitKey(-1)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Read images
images = glob.glob('/home/pi/Desktop/calibration/test*.jpg')
ret = []
corners = []

for fname in images:
    img = cv2.imread(fname)
    ret, corners = FindAndDisplayChessboard(img)
    print (corners)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)


img = cv2.imread(fname)
#print(img.shape[1::-1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#print(gray.shape[::-1])

height, width , channels = img.shape
s = (width,height)
print(s)

ret, intrinsics, distortion, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width,height) , None, None)

# Show results in console
print("Intrinsics: ")
print (intrinsics)
print("Distortion : ")
print(distortion)
for i in range(len(tvecs)):
    print ("Translations(%d) : " % i )
    print(tvecs[0])
    print ("Rotation(%d) : " % i )
    print(rvecs[0])

# print to file
from tempfile import TemporaryFile
outfile = open('camera.npz','w')
np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)

npload = np.load('camera.npz')
print(npload['intrinsics'])
print(npload['distortion'])

cv2.waitKey(-1)
cv2.destroyAllWindows()

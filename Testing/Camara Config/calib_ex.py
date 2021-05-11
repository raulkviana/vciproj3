import numpy as np
import cv2
import glob

# Board Size
board_w = 8;
board_h = 6;

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


def  FindAndDisplayChessboard(img):
    # Find the chess board corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape[::-1])
    r, c = cv2.findChessboardCorners(gray, (board_w,board_h), None)

    # If found, display image with corners
    if r == True:
        img = cv2.drawChessboardCorners(img, (board_w, board_h), c, r)
        cv2.imshow('img',img)
        #cv2.waitKey(1)

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


# for fname in images:
#     img = cv2.imread(fname)
#     ret, corners = FindAndDisplayChessboard(img)
#     print (corners)
#     if ret == True:
#         objpoints.append(objp)
#         imgpoints.append(corners)

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print("Could not open video device")

# To set the resolution
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    ret, corners = FindAndDisplayChessboard(frame)
    print (corners)
    if ret == True:
        img = frame
        objpoints.append(objp)
        imgpoints.append(corners)

    # Waits for a user input to quit the application
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
#print(img.shape[1::-1])
# img = cv2.imread(fname)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray.shape[::-1])

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
    # print ("Rotation(%d) : " % i )
    # print(rvecs[0])

# print to file
from tempfile import TemporaryFile
#outfile = open('camera.npz','w')
np.savez('camera.npz', intrinsics=intrinsics, distortion=distortion)

npload = np.load('camera.npz')
print(npload['intrinsics'])
print(npload['distortion'])

cv2.waitKey(-1)

Read and display Line
#fname=images[0]
#img = cv2.imread(fname)

normal
normal = np.float32([[0,0,0], [0,0,-1]]).reshape(-1,3)
imgpts, jac = cv2.projectPoints(normal, rvecs[0], tvecs[0], intrinsics, distortion)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), 5)
#
cv2.imshow('img',img)
cv2.waitKey(-1)

axis
axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
imgpts, jac = cv2.projectPoints(axis, rvecs[0], tvecs[0], intrinsics, distortion)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), 5)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255,0,0), 5)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (0,255,0), 5)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 5)
#
cv2.imshow('img',img)
cv2.waitKey(-1)

cube
cube = np.float32([[0,0,0], [0,0,-1], [1,0,-1], [1,0,0], [0,1,0], [0,1,-1], [1,1,-1], [1,1,0]]).reshape(-1,3)
imgpts, jac = cv2.projectPoints(cube, rvecs[len(rvecs)-1], tvecs[len(rvecs)-1], intrinsics, distortion)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[4].ravel()), tuple(imgpts[5].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[5].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[6].ravel()), tuple(imgpts[7].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[7].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[4].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[5].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[6].ravel()), (0,0,255), 2)
img = cv2.line(img, tuple(imgpts[3].ravel()), tuple(imgpts[7].ravel()), (0,0,255), 2)
#
cv2.imshow('img',img)
cv2.waitKey(-1)



cv2.destroyAllWindows()

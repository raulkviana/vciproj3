#finding hsv range of target object(pen)
import cv2 as cv
import numpy as np
import time
# A required callback method that goes into the trackbar function.
def nothing(x):
    pass

# import the image
frame = cv.imread("/home/alegria/VCI/OPENCV_COURSE/lego_red.jpg")
frame_contours = frame.copy()
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)



# Convert the BGR image to HSV image.

l_red = [0,133,112]
u_red = [179,255,255]

# Set the lower and upper HSV range according to the value selected
# by the trackbar
lower_range = np.array(l_red)
upper_range = np.array(u_red)

# Filter the image and get the binary mask, where white represents 
# your target color
mask = cv.inRange(hsv, lower_range, upper_range)

# You can also visualize the real part of the target color (Optional)
res = cv.bitwise_and(frame, frame, mask=mask)

# Converting the binary mask to 3 channel image, this is just so 
# we can stack it with the others
mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

# Morphological operation -> closing
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel)

# Find edges
masked_new = cv.bitwise_and(frame,frame,mask = closing)
# edge finding for the new image with closing
gray = cv.cvtColor(masked_new,cv.COLOR_BGR2GRAY)
edge = cv.Canny(gray,100,200)
# edge finding for the original image
gray_2 = cv.cvtColor(res,cv.COLOR_BGR2GRAY)
edge2 = cv.Canny(gray_2,100,200)


# contours
contours, hierarchy = cv.findContours(edge, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[-2:]
cv.drawContours(frame_contours,contours,-1,(0,255),3)

# stack
# edge finding
stacked = np.hstack((edge,edge2))
# Morfological, Edges  operations
stacked_2 = np.hstack((closing,edge))
# initial photos
stacked_3 = np.hstack((mask_3,frame,res))

# Show this stacked frames
cv.imshow('Edge finding',cv.resize(stacked,None,fx=0.5,fy=0.5))
cv.imshow('Contours finding and draw',cv.resize(frame_contours,None,fx=0.5,fy=0.5))
cv.imshow('Morfological, Edges operations',cv.resize(stacked_2,None,fx=0.4,fy=0.4))
cv.imshow('Initial Images',cv.resize(stacked_3,None,fx=0.4,fy=0.4))


# If the user presses ESC then exit the program
key = cv.waitKey(0)
if key == 27:
    cv.destroyAllWindows()


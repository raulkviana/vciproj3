"""@package color trackbar program
Documentation for this module.

This program as the objective of identifying colors(HSV) in an image using trackbars and the mouse coursor.
"""
import cv2 as cv
import numpy as np
import time


def nothing(x):
    """
    A required callback method that goes into the trackbar function.
    """
    pass

def resize(img,fx,fy):
    """
    resize function
    @param [in]
    @param [in]
    @param [out]
    """
    height, width = img.shape[:2]
    size = (int(width * fx), int(height * fy))  # bgr
    img = cv.resize(img, size)

    return img

def getposHsv(event,x,y,flags,param):
    """
    function to transform the event from the mouse into a variation in the image and trackbar. 
    The variation is [-10,+10] from the original value.
    @param [in] event
    @param [in] x : x value returned from the mouse click
    @param [in] y : y value returned from the mouse click
    @param [in] param
    """

    global hsv_ret,color_h,color_s,color_v 
    global l_h,u_h,l_v,u_v,u_s,l_s
    
    if event==cv.EVENT_LBUTTONDOWN:
        hsv_ret = hsv[y,x]
        color_h = hsv[y,x,0]
        color_s = hsv[y,x,1]
        color_v = hsv[y,x,2]
        l_h = color_h - 10
        u_h = color_h + 10
        l_s = color_s - 10
        u_s = color_s + 10
        l_v = color_v - 10
        u_v = color_v + 10


        l_h = cv.setTrackbarPos("L - H", "Trackbars",l_h)
        l_s = cv.setTrackbarPos("L - S", "Trackbars",l_s)
        l_v = cv.setTrackbarPos("L - V", "Trackbars",l_v)
        u_h = cv.setTrackbarPos("U - H", "Trackbars",u_h)
        u_s = cv.setTrackbarPos("U - S", "Trackbars",u_s)
        u_v = cv.setTrackbarPos("U - V", "Trackbars",u_v)


# import the image
frame = cv.imread("/home/alegria/VCI/OPENCV_COURSE/vciproj3/dataset2/rect/IMG_20210412_173344.jpg")

# resize function
frame = resize(frame,0.2,0.2)

# Create a window named trackbars.
cv.namedWindow("Trackbars")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

# Convert the BGR image to HSV image.
cv.setMouseCallback("Trackbars",getposHsv)

while True:

    # Convert the BGR image to HSV image.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Get the new values of the trackbar in real time as the user changes 
    # them
    l_h = cv.getTrackbarPos("L - H", "Trackbars")
    l_s = cv.getTrackbarPos("L - S", "Trackbars")
    l_v = cv.getTrackbarPos("L - V", "Trackbars")
    u_h = cv.getTrackbarPos("U - H", "Trackbars")
    u_s = cv.getTrackbarPos("U - S", "Trackbars")
    u_v = cv.getTrackbarPos("U - V", "Trackbars")
 
    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array([l_h, l_s, l_v])
    upper_range = np.array([u_h, u_s, u_v])
    
    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv.inRange(hsv, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    res = cv.bitwise_and(frame, frame, mask=mask)


    # load background (could be an image too)
    bk = np.full(frame.shape, 255, dtype=np.uint8)  # white bk

    # get masked background, mask must be inverted 
    mask = cv.bitwise_not(mask)
    bk_masked = cv.bitwise_and(bk, bk, mask=mask)

    # combine masked foreground and masked background 
    final = cv.bitwise_or(res, bk_masked)
    
    mask = cv.bitwise_not(mask)  # revert mask to original


    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    
    # stack the mask, orginal frame and the filtered result
    stacked = np.hstack((mask_3,res,final))
    
    # Show this stacked frame at 40% of the size.
    cv.imshow("Trackbars",frame)
    cv.imshow("stacked",stacked)

    # If the user presses ESC then exit the program
    key = cv.waitKey(1)
    if key == 27:
        break

    
    # # If the user presses `s` then print this array.
    # if key == ord('s'):
        
    #     thearray = [[l_h,l_s,l_v],[u_h, u_s, u_v]]
    #     print(thearray)
        
    #     # Also save this array as penval.npy
    #     np.savetxt('hsv_value',thearray)
    #     break

        
    
cv.destroyAllWindows()
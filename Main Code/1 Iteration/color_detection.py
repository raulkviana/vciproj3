"""@package color detection rect
Documentation for this module.

This program as the objective of identifying only rect pieces based on their predefined HSV(Dictionary).
"""
import cv2 as cv
import numpy as np
import time
# in order to read the color json file
import json

#Letter font
font = cv.FONT_HERSHEY_SIMPLEX


"""
Functions
"""
def resize(img,fx,fy):
    height, width = img.shape[:2]
    size = (int(width * fx), int(height * fy))  # bgr
    img = cv.resize(img, size)

    return img

def imageprocessing(frame,hsv_low,hsv_upper):
    """
    Used to mask the desired region using HSV range
    @param [in] frame : input image
    @param [in] hsv_low : HSV lower range
    @param [in] hsv_upper: HSV upper range

    @param [out] mask_3 : masks the image in order to obtain the filter piece
    @param [out] closing : morphological operation
    @param [out] color_piece : visualize the real part of the target 
    """

    # Convert the BGR image to HSV image.
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Set the lower and upper HSV range according to the value selected
    # by the trackbar
    lower_range = np.array(hsv_low)
    upper_range = np.array(hsv_upper)

    # Filter the image and get the binary mask, where white represents 
    # your target color
    mask = cv.inRange(hsv, lower_range, upper_range)

    # You can also visualize the real part of the target color (Optional)
    color_piece = cv.bitwise_and(frame, frame, mask=mask)

    # Converting the binary mask to 3 channel image, this is just so 
    # we can stack it with the others
    mask_3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    # Morphological operation -> closing
    kernel = np.ones((10,10),np.uint8)
    closing = cv.morphologyEx(mask,cv.MORPH_CLOSE,kernel,iterations = 10)

    return closing, color_piece, mask_3

def edge_finding(frame,closing,color_piece):
    """
    Edge finding function
    @param [in] frame : input image
    @param [in] closing : morphological operation
    @param [in] color_piece : visualize the real part of the target 

    @param [out] edge : edge finding for the original image 
    @param [out] edge_color: edge finding in the color piece 
    """

    # Find edges
    masked_new = cv.bitwise_and(frame,frame,mask = closing)
    # edge finding for the new image with closing
    gray = cv.cvtColor(masked_new,cv.COLOR_BGR2GRAY)
    # edge finding for the original image 
    edge = cv.Canny(gray,100,200)
    #edge finding in the color piece 
    gray_2 = cv.cvtColor(color_piece,cv.COLOR_BGR2GRAY)
    edge_color = cv.Canny(gray_2,100,200)

    return edge, edge_color

def name_contour(closing,frame_contours):
    """
    Contour function
    @param [in] closing : morphological operation
    @param [in] frame_contours : image copy

    @param [out] frame_countours : contours output
    """
    contours, hierarchy = cv.findContours(closing, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2:]
    for c in contours:

        approx = cv.approxPolyDP(c, 0.01*cv.arcLength(c, True), True)
        print("\napprox: ",approx)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        if ( closing.shape[1]*closing.shape[0]/300 <cv.contourArea(approx) < closing.shape[1]*closing.shape[0]/4 ):
            if len(approx) == 4  :
                x, y, w, h = cv.boundingRect(approx)
                aspect_ratio = float(w) / float(h)
                print("aspect:",aspect_ratio)
                if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                    cv.putText(frame_contours, hsv_color + "Square", (x, y),font,3,(0,255,0),5)
                    cv.rectangle(frame_contours,(x,y),(x+w,y+h),(0,255,0),10)
                else:
                    cv.putText(frame_contours, hsv_color + "Rectangle", (x, y),font,3,(0,255,0),5)
                    cv.rectangle(frame_contours,(x,y),(x+w,y+h),(0,255,0),10)
            else:
                cv.putText(frame_contours, hsv_color + " Circle", (x, y),font,3,(0,255,0),5)
                cv.drawContours(frame_contours, [approx], 0, (0, 255, 0), 5)

    return frame_contours

"""
    Main Program
"""

# import the image
frame = cv.imread("/home/alegria/VCI/OPENCV_COURSE/vciproj3/dataset2/rect/IMG_20210412_173344.jpg")
#frame = cv.imread("/home/alegria/VCI/OPENCV_COURSE/vciproj3/dataset2/rect_and_nonrect/IMG_20210412_174406.jpg")
frame_contours = frame.copy()



# #read file
# with open('/home/alegria/VCI/OPENCV_COURSE/Color_detection/color_nonrect.json') as json_file:
#     data = json_file.read()
with open('/home/alegria/VCI/OPENCV_COURSE/colors.json') as json_file:
   data = json_file.read()

# parse file
obj = json.loads(data)

for f in obj['colors']:

    hsv_color = (f['color'])
    hsv_low = (f['hsv_low'])
    hsv_upper =(f['hsv_upper'])

    closing,color_piece, mask_3 = imageprocessing(frame,hsv_low,hsv_upper)
    edge,edge_color = edge_finding(frame,closing,color_piece)
    frame_contours = name_contour(closing,frame_contours)

    ## stack
    # edge finding
    stacked = np.hstack((edge,edge_color))
    # Morfological, Edges  operations
    stacked_2 = np.hstack((closing,edge))
    # initial photos
    stacked_3 = np.hstack((mask_3,frame,color_piece))

    # Show this stacked frames
    cv.imshow('Edge finding',cv.resize(stacked,None,fx=0.2,fy=0.2))
    cv.imshow('Morfological, Edges operations',cv.resize(stacked_2,None,fx=0.2,fy=0.2))
    cv.imshow('Initial Images',cv.resize(stacked_3,None,fx=0.2,fy=0.2))
    cv.imshow('Contours finding and draw',cv.resize(frame_contours,None,fx=0.2,fy=0.2))



# If the user presses ESC then exit the program
key = cv.waitKey(0)
if key == 27:
    cv.destroyAllWindows()


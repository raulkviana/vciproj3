import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import sys, os
import glob
sys.path.append(os.path.abspath(os.path.join('..', 'dataset_iterator')))
import dataset_iterator

windowNameConfig = 'corners'
default_upper_value = 200

def change_file(x):
    pass

def process_image(img):

    # Turn to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscaling',gray)

    # Equalizing
    clahe = cv.createCLAHE(clipLimit=3.0)
    equ = clahe.apply(gray) # Comportamento muito bom pra imagens diferentes
    #equ = cv.equalizeHist(gray) # Ficou a quem das espectativas
    cv.imshow('Equalizing',equ)

    sift = cv.SIFT_create()
    kp = sift.detect(equ, None)
    dst = cv.drawKeypoints(equ, kp, img)
    cv.imshow('SIFT', dst) # Heaven sent!

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(equ, None)
    fast.setNonmaxSuppression(0)
    cv.imshow('FAST',cv.drawKeypoints(equ, kp, None, color=(255, 0, 0)))

    # Initiate FAST detector
    star = cv.xfeatures2d.StarDetector_create()
    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    # find the keypoints with STAR
    kp = star.detect(equ, None)
    # compute the descriptors with BRIEF
    kp, des = brief.compute(equ, kp)
    cv.imshow('BRIEF',cv.drawKeypoints(equ, kp, None, color=(255, 0, 0)))

    # Blur
    if 0 == cv.getTrackbarPos('Bluring', windowNameConfig) % 2:
        ksizeBlur = cv.getTrackbarPos('Bluring', windowNameConfig) + 1
    else:
        ksizeBlur = cv.getTrackbarPos('Bluring', windowNameConfig)
    blur = cv.GaussianBlur(equ, (ksizeBlur, ksizeBlur), 0) # Um bom valor é 10 para o kernel: 10 x 10
    #blur = cv.medianBlur(equ, ksizeBlur) # Nao parace ser uma boa opcao
    #blur = cv.bilateralFilter(equ, 5, ksizeBlur, ksizeBlur)
    cv.imshow('Blur', blur)

    # Harris corner detection
    blur = np.float32(blur)
    dst = cv.cornerHarris(blur, 2, 3, 0.04)
    cv.imshow('Harris',dst)

    # Erode
    kernelOP = cv.getTrackbarPos('Opening Kernel', windowNameConfig)  # Bom valor is 1
    erosion = cv.erode(dst, np.ones((kernelOP, kernelOP), np.uint8))
    cv.imshow('Erosion', erosion)

    # Closing
    kernelCl = cv.getTrackbarPos('Closing Kernel', windowNameConfig) # Bom valor is 7
    close = cv.morphologyEx(erosion, cv.MORPH_CLOSE, np.ones((kernelCl, kernelCl), np.uint8))
    cv.imshow('Closing', close)

    # Threshold for an optimal value, it may vary depending on the image.
    const = 0.0001 # Constant to get the minimum value for Thresholding
    img[close > const * close.max()] = [0, 0, 255]

    return img

pics = glob.glob('../dataset_iterator/dataset/' + '*.jpg')

#Create trackbar for iterating and changing algorithm parameters
cv.namedWindow(windowNameConfig)
cv.createTrackbar('File',windowNameConfig,0,len(pics)-1,change_file)
cv.createTrackbar('Bluring',windowNameConfig,1,default_upper_value,change_file)
cv.createTrackbar('Closing Kernel',windowNameConfig,1,10,change_file)
cv.createTrackbar('Opening Kernel',windowNameConfig,1,10,change_file)

while(1):
    picPos = cv.getTrackbarPos('File', windowNameConfig)
    filename = pics[picPos]

    # Read Image
    img = cv.imread(filename)

    # Resize image
    imgRe = dataset_iterator.resize(img, 10)
    cv.imshow('Original Image ', imgRe)

    finalIma = process_image(imgRe)

    cv.imshow(windowNameConfig, finalIma)

    if cv.waitKey(1) & 0xff == 27:
        cv.destroyAllWindows()
        break
import glob
import cv2 as cv
import numpy as np

directory_str = 'dataset/'
# directory_str = '../dataset2/rect/'
scale_percent = 20


def resize(img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


for file in glob.glob(directory_str + '*.jpg'):
    # resizing window
    img = cv.imread(file)
    img_resized = resize(img)
    cv.imshow('resized', img_resized)
    # convert to grayscale
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    #cv.imshow('gray', img_gray)

    # histogram equalizer -> enhance the contrast
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
    img_equ = clahe.apply(img_gray)

    # Blurring
    kernel_blur_size = 3
    img_blur = cv.GaussianBlur(img_gray, (kernel_blur_size, kernel_blur_size), 0)
    # cv.imshow('Gaussian Blur', img_blur)

    # Sobel x
    kernel_filter_size = 3
    img_sobel_x = cv.Sobel(img_blur, cv.CV_64F, 1, 0, ksize=kernel_filter_size)
    # cv.imshow('sobel_x', img_sobel_x)
    img_sobel_x = np.uint8(np.absolute(img_sobel_x))

    # Sobel y
    img_sobel_y = cv.Sobel(img_blur, cv.CV_64F, 0, 1, ksize=kernel_filter_size)
    #cv.imshow('sobel_y', img_sobel_y)
    img_sobel_y = np.uint8(np.absolute(img_sobel_y))

    # thresholding
    img_thresh_x = cv.adaptiveThreshold(img_sobel_x, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2)
    #cv.imshow('Adaptative Threshold X', img_thresh_x)

    img_thresh_y = cv.adaptiveThreshold(img_sobel_y, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2)
    #cv.imshow('Adaptative Threshold Y', img_thresh_y)

    # Morphological operators

    # Erode
    kernel_erode_size = 7
    kernel_erode = np.ones((kernel_erode_size, kernel_filter_size), np.uint8)
    img_erosion_x = cv.erode(img_thresh_x, kernel_erode, iterations=1)
    img_erosion_y = cv.erode(img_thresh_y, kernel_erode, iterations=1)
    # cv.imshow('Erosion X', img_erosion_x)
    # cv.imshow('Erosion Y', img_erosion_y)

    # Dilation
    kernel_dilation_size = 3
    kernel_dilation = np.ones((kernel_dilation_size, kernel_dilation_size), np.uint8)
    img_dilation_x = cv.dilate(img_erosion_x, kernel_dilation, iterations=1)
    img_dilation_y = cv.dilate(img_erosion_y, kernel_dilation, iterations=1)
    cv.imshow('Dilation X', img_dilation_x)
    cv.imshow('Dilation Y', img_dilation_y)

    # contours
    contours, hierarchy = cv.findContours(img_dilation_x, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_resized, contours, -1, (255, 0, 0), 1)

    k = cv.waitKey(0)
    if k == ord('q'):
        break

cv.destroyAllWindows()

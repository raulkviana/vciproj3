import glob
import cv2 as cv
import numpy as np

directory_str = 'dataset/'
#directory_str = '../dataset2/rect/'

scale_percent = 20

kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.uint8)


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

    # blurring
    kernel_blur_size = 5
    img_blur = cv.GaussianBlur(img_gray, (kernel_blur_size, kernel_blur_size), 0)
    # cv.imshow('Gaussian Blur', img_blur)

    # laplacian filter
    kernel_lapl_size = 3
    img_laplacian = cv.Laplacian(img_blur, cv.CV_64F, ksize=kernel_lapl_size)
    cv.imshow('laplacian', img_laplacian)
    img_laplacian = np.uint8(np.absolute(img_laplacian))

    # thresholding
    img_thresh = cv.adaptiveThreshold(img_laplacian, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 2)
    cv.imshow('Adaptative Threshold', img_thresh)

    # Morphological operators
    # Opening
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_opening = cv.morphologyEx(img_thresh, cv.MORPH_OPEN, kernel, iterations=1)
    cv.imshow('Opening', img_opening)

    # contours
    contours, hierarchy = cv.findContours(img_opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(img_resized, contours, -1, (0, 255, 0), 1)
    #cv.imshow('resized', img_resized)

    k = cv.waitKey(0)
    if k == ord('q'):
        break

cv.destroyAllWindows()

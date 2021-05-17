import cv2 as cv
import numpy as np
from statistics import mean
from time import sleep

""" Global Variables """
mouse_yx = []


def read_img(file_name, scale_percent):
    """ function that reads and resizes the image """
    img = cv.imread(file_name)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def on_mouse(event, x, y, flags, params):
    """ mouse callback function that appends the (y,x) coordinates of the mouse click to the global list """
    if event == cv.EVENT_LBUTTONDOWN:
        print('(x,y) = ({},{})'.format(x, y))
        mouse_yx.append((y, x))


def sum_abs_diffs(hsv1, hsv2):
    """
    function to compute the sum of absolute differences between 2 HSV values

    @param hsv1 : numpy array with the HSV values
    @param hsv2 : numpy array with the HSV values

    @returns : sum of absolute differences between 2 HSV values
    """
    return abs(int(hsv1[0]) - int(hsv2[0])) + abs(int(hsv1[1]) - int(hsv2[1])) + abs(int(hsv1[2]) - int(hsv2[2]))


def sum_diffs(hsv1, hsv2):
    """
    function to compute the sum of differences between 2 HSV values

    @param hsv1 : numpy array with the HSV values
    @param hsv2 : numpy array with the HSV values

    @returns : sum of differences between 2 HSV values
    """
    return (int(hsv1[0]) - int(hsv2[0])) + (int(hsv1[1]) - int(hsv2[1])) + (int(hsv1[2]) - int(hsv2[2]))


def update_hsv_values(ref_hsv, lst_new_hsv, hsv_min_max):
    """
    function to update the min and max HSV values and the HSV reference value for the Region Growing

    @param ref_hsv : list with the HSV reference value
    @param lst_new_hsv : list of lists with the HSV values of the new added pixels
    @param hsv_min_max : list with 2 lists containing the current min and max HSV values

    @returns ref_hsv : list with the updated HSV reference value
    @returns hsv_min_max : list with 2 lists with the updated min and max HSV values
    """
    for hsv in lst_new_hsv:
        if sum_diffs(hsv, hsv_min_max[0]) < 0:
            hsv_min_max[0] = hsv
        else:
            hsv_min_max[1] = hsv

    for i in range(3):
        ref_hsv[i] = int((mean([int(val[i]) for val in lst_new_hsv]) + ref_hsv[i]) / 2)

    return ref_hsv, hsv_min_max


def region_growing4_hsv(img, init_seed, threshold):
    """
    function to implement region growing algorithm (4 neighbors version)
    this version updates de HSV reference value used in the comparison criteria dynamically
    with a moving mean method

    @param img : HSV image to perform the algorithm
    @param init_seed : initial seed (x,y) point
    @param threshold : threshold used for the comparison criteria

    @returns matrix : HSV image with the region extracted
    @returns min_max_hsv : minimum and maximum HSV values in the region
    """
    height, width, _ = img.shape
    all_ones = np.iinfo(np.uint8).max
    region = np.array([all_ones, all_ones, all_ones])
    matrix = np.zeros([height, width, 3], np.uint8)
    """ 4 neighboring pixels implementation """
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    """ get the initial seed's HSV values and add the seed to the region """
    ref_hsv = img[init_seed][:]
    print("initial seed HSV: ", ref_hsv)
    matrix[init_seed][:] = region
    min_max_hsv = [ref_hsv, ref_hsv]
    """ initialize the stack with the initial seed """
    stack = [init_seed]
    """ use number of iterations to implement a stop condition """
    it = 0
    cv.namedWindow('Region Growing Demo')
    while stack:
        for pix in stack:
            for i in range(len(neighbors)):
                """ compute the position of neighbor pixels """
                y_new = pix[0] + neighbors[i][0]
                x_new = pix[1] + neighbors[i][1]
                new_pix = (y_new, x_new)
                lst_new_hsv = []

                it += 1
                """ check if new pixel is inside the image """
                check_inside = (x_new >= 0) and (y_new >= 0) and (x_new < width) and (y_new < height)
                if check_inside:
                    new_hsv = img[new_pix][:]
                    """ condition to check if new pix meets the threshold """
                    thresh_condition = sum_abs_diffs(ref_hsv, new_hsv) < threshold
                    """ check if new pixel already belongs to the region """
                    pixel_belongs = np.all(np.equal(matrix[new_pix][:], region))
                    if thresh_condition and not pixel_belongs:
                        stack.append(new_pix)
                        lst_new_hsv.append(img[new_pix][:])
                        matrix[new_pix][:] = region
                        img_demo_hsv = np.bitwise_and(img, np.invert(matrix))
                        img_demo = cv.cvtColor(img_demo_hsv, cv.COLOR_HSV2RGB)
                        cv.circle(img_demo, (init_seed[1], init_seed[0]), 3, (170, 90, 200), 3)
                        cv.imshow('Region Growing Demo', img_demo)
                        cv.waitKey(1)

            stack.remove(pix)
            """ update the HSV values only if new pixels were added """
            if lst_new_hsv:
                ref_hsv, min_max_hsv = update_hsv_values(ref_hsv, lst_new_hsv, min_max_hsv)
            """ stop condition """

    print("iterations = ", it)
    cv.waitKey(0)
    return matrix, min_max_hsv


def nothing(x):
    pass


def main():
    """ local variables"""
    # file_name = 'color_test.png'
    scale_percent = 50
    file_name = '../dataset2/rect/color_test.jpg'
    default_thresh = 50
    img = read_img(file_name, scale_percent)
    height, width, _ = img.shape
    """ display the dataset's first image """
    cv.imshow('Original', img)
    cv.createTrackbar('threshold', 'Original', default_thresh, 255, nothing)

    """ convert image to HSV to feed it to the Region Growing algorithm """
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)

    while True:
        cv.setMouseCallback('Original', on_mouse, 0, )

        k = cv.waitKey()
        """ quits the program """
        if k == ord('q'):
            break
        """ iterates through the the dataset's image """

        """ region growing """
        if k == ord('g') and mouse_yx:
            """ use the position of the last click as the seed"""
            seed = mouse_yx[-1]

            threshold = cv.getTrackbarPos('threshold', 'Original')  # 50 not bad
            new_mask, min_max_hsv = region_growing4_hsv(img_hsv, seed, threshold)

            print("min HSV: {}\tmax HSV: {}".format(min_max_hsv[0], min_max_hsv[1]))

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

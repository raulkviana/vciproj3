import cv2 as cv
import numpy as np
import glob
import pprint
import hsv
import json

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
    max_it = height * width / 10
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
                    thresh_condition = hsv.sum_abs_diffs(ref_hsv, new_hsv) < threshold
                    """ check if new pixel already belongs to the region """
                    pixel_belongs = np.all(np.equal(matrix[new_pix][:], region))
                    if thresh_condition and not pixel_belongs:
                        stack.append(new_pix)
                        lst_new_hsv.append(img[new_pix][:])
                        matrix[new_pix][:] = region

            stack.remove(pix)
            """ update the HSV values only if new pixels were added """
            if lst_new_hsv:
                ref_hsv = hsv.update_hsv_ref(lst_new_hsv, ref_hsv)
                min_max_hsv = hsv.update_hsv_min_max(lst_new_hsv, min_max_hsv)

            """ stop condition to avoid the algorithm to diverge """
            if it > max_it:
                break

    print("iterations = ", it)
    return matrix, min_max_hsv


def nothing(x):
    pass


def main():
    """ local variables"""
    scale_percent = 100
    dir_str = '../datasetBox/'
    default_thresh = 50
    """ create an iterator for the dataset """
    lst_img = glob.glob(dir_str + '*.jpg')
    img_iter = iter(lst_img)
    img = read_img(next(img_iter), scale_percent)
    """ UNCOMMENT LINES ABOVE TO TEST A SPECIFIC IMAGE """
    #scale_percent = 100
    #filename = "../dataset2/yellow.png"
    #img = read_img(filename, scale_percent)
    height, width, _ = img.shape
    """ DISPLAY DATASET'S FIRST IMAGE """
    cv.imshow('Original', img)
    cv.createTrackbar('threshold', 'Original', default_thresh, 255, nothing)
    cv.imshow('Region Growing', img)
    """ convert image to HSV to feed it to the Region Growing algorithm """
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    mask = old_mask = None
    min_max_hsv = None
    dict_hsv = {}
    while True:
        cv.setMouseCallback('Original', on_mouse, 0, )

        k = cv.waitKey()
        """ quits the program """
        if k == ord('q'):
            break
        """ iterates through the the dataset's image """
        if k == ord('n'):
            file_name = next(img_iter, None)
            if file_name is None:
                print("Reached dataset's end!")
                break

            """ destroy previous images """
            cv.destroyWindow('Original')
            cv.destroyWindow('Region Growing')
            """ read and display new image """
            img = read_img(file_name, scale_percent)
            height, width, _ = img.shape
            img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            cv.imshow('Original', img)
            cv.imshow('Region Growing', img)
            cv.createTrackbar('threshold', 'Original', default_thresh, 255, nothing)
            """ reset mask"""
            mask = old_mask = np.zeros([height, width, 3], np.uint8)

        """ resets image """
        if k == ord('r'):
            """ destroy previous images """
            cv.destroyWindow('Original')
            cv.destroyWindow('Region Growing')
            """ display the same images """
            cv.imshow('Original', img)
            cv.imshow('Region Growing', img)
            cv.createTrackbar('threshold', 'Original', default_thresh, 255, nothing)
            """ reset mask """
            mask = old_mask = np.zeros([height, width, 3], np.uint8)

        """ region growing """
        if k == ord('g') and mouse_yx:
            """ use the position of the last click as the seed"""
            seed = mouse_yx[-1]
            threshold = cv.getTrackbarPos('threshold', 'Original')  # 50 shows a good performance overall
            print("\n----------- NEW ITERATION ----------- \n")
            new_mask, min_max_hsv = region_growing4_hsv(img_hsv, seed, threshold)
            if mask is None:
                mask = new_mask
            else:
                mask = np.bitwise_or(new_mask, old_mask)

            img_region_growing_hsv = np.bitwise_and(img_hsv, np.invert(mask))

            print("min HSV: {}\tmax HSV: {}".format(min_max_hsv[0], min_max_hsv[1]))
            """ display region growing and highlight the seed location """
            img_region_growing_bgr = cv.cvtColor(img_region_growing_hsv, cv.COLOR_HSV2BGR)
            cv.circle(img_region_growing_bgr, (seed[1], seed[0]), 3, (170, 90, 200), 3)
            cv.imshow('Region Growing', img_region_growing_bgr)
            """ save the last mask to allow mask overlapping """
            old_mask = np.copy(mask)

        " save current min and max HSV values to dictionary "
        if k == ord('s') and min_max_hsv.any():
            yn = input("Add current min_max_HSV to dict? (y/n): ")
            if yn == 'y':
                hsv.update_dict(dict_hsv, min_max_hsv)

        """ print current dictionary """
        if k == ord('p') and dict_hsv:
            print(pprint.pformat(dict_hsv))

        """ write dictionary to a JSON file """
        if k == ord('w'):
            if dict_hsv:
                hsv_json = json.dumps(dict_hsv)
                json_file_name = input("file name (without extension): ")
                with open(json_file_name + ".txt", mode='w') as json_file:
                    json.dump(hsv_json, json_file, indent=4)
            else:
                print("Empty dictionary, add more colors!")

    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

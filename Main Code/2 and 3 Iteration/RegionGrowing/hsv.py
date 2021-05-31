import numpy as np
from statistics import mean

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


def update_hsv_min_max(lst_new_hsv, hsv_min_max):
    """
    function to update the min and max HSV values

    @param lst_new_hsv : list of lists with the HSV values of the new added pixels
    @param hsv_min_max : list with 2 lists containing the current min and max HSV values

    @returns hsv_min_max : list with 2 lists with the updated min and max HSV values
    """
    new_hsv_min_max = np.copy(hsv_min_max)
    for hsv_val in lst_new_hsv:
        for i in range(3):
            if hsv_val[i] < new_hsv_min_max[0][i]:
                new_hsv_min_max[0][i] = hsv_val[i]
            if hsv_val[i] > new_hsv_min_max[1][i]:
                new_hsv_min_max[1][i] = hsv_val[i]

    return new_hsv_min_max


def update_hsv_ref(lst_new_hsv, ref_hsv):
    """
    function to update the HSV reference value for the Region Growing

    @param ref_hsv : list with the HSV reference value
    @param lst_new_hsv : list of lists with the HSV values of the new added pixels

    @returns ref_hsv : list with the updated HSV reference value
    """
    for i in range(3):
        ref_hsv[i] = int((mean([int(val[i]) for val in lst_new_hsv]) + ref_hsv[i]) / 2)

    return ref_hsv


def update_dict(dict_hsv, min_max_hsv):
    """
    function to add new colors or update the dictionary's min and max HSV values

    @param dict_hsv : dict with keys -> colors names and values -> list of 2 lists containing min and max HSV values
    @param new_min_max_hsv : list of 2 lists containing the new min and max HSV values to add/update the dictionary

    @return : None
    """

    color_name = input("Color name: ")
    if color_name in dict_hsv:
        """ update min HSV value """
        new_min_max_hsv = update_hsv_min_max([hsv_val.tolist() for hsv_val in min_max_hsv], dict_hsv[color_name])
        dict_hsv[color_name] = new_min_max_hsv.tolist()

    else:
        """ convert list of np.arrays to list of lists and add it to the colors dictionary"""
        dict_hsv[color_name] = [hsv_val.tolist() for hsv_val in min_max_hsv]
        print("Added {color} color to dictionary!".format(color=color_name))

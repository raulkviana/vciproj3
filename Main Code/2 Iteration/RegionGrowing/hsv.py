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
    for hsv_val in lst_new_hsv:

        if sum_diffs(hsv_val, hsv_min_max[0]) < 0:
            hsv_min_max[0] = hsv_val

        if sum_diffs(hsv_val, hsv_min_max[1]) > 0:
            hsv_min_max[1] = hsv_val

    return hsv_min_max


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


def update_dict(dict_hsv, new_min_max_hsv):
    """
    function to add new colors or update the dictionary's min and max HSV values

    @param dict_hsv : dict with keys -> colors names and values -> list of 2 lists containing min and max HSV values
    @param new_min_max_hsv : list of 2 lists containing the new min and max HSV values to add/update the dictionary

    @return : None
    """

    color_name = input("Color name: ")
    if color_name in dict_hsv:
        """ update min HSV value """
        if sum_diffs(new_min_max_hsv[0], dict_hsv[color_name][0]) < 0:
            dict_hsv[color_name][0] = new_min_max_hsv[0].tolist()
            print("{color} HSV min value updated!".format(color=color_name))

        """ update max HSV value """
        if sum_diffs(new_min_max_hsv[1], dict_hsv[color_name][1]) > 0:
            dict_hsv[color_name][1] = new_min_max_hsv[1].tolist()
            print("{color} HSV max value updated!".format(color=color_name))

    else:
        """ convert list of np.arrays to list of lists and add it to the colors dictionary"""
        dict_hsv[color_name] = [hsv_val.tolist() for hsv_val in new_min_max_hsv]
        print("Added {color} color to dictionary!".format(color=color_name))

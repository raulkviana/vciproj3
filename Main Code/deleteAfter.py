import numpy as np
import cv2 as cv
from Constants import constants_app
from Constants import constants_feat
from calib import Calibration
from lego import Lego
from feature_extrac import FeatureExtrac as fe
from time import sleep
import csv


def compile2save(lst):
    out_lst = []
    for elem in lst:
        temp = []

        # Append color and ratio
        str1 = elem.color +'/'+ str(elem.ratio[0]) + 'x' + str(elem.ratio[1])
        temp.append(str1)

        # Append if it is rect or non rect
        if elem.rect:
            temp.append('rect')
        else:
            temp.append('non-rect')

        # Append contour
        temp.append(elem.contour.tolist())

        # Append to output list
        out_lst.append(temp)

    print(out_lst)

    return out_lst

def write2file(name):
    with open(name + '.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(constants_app.FIELDS_OUTPUT_FILE)
        # Write legos to csv file
        write.writerows(compile2save(lego_lst))


if __name__ == "__main__":
    # App module starting
    print('Starting lego search !!!')
    print('\n\nInstructions:')
    print('Press s to save legos to file')
    print('Press q to quit the app')

    calib = Calibration()
    fe = fe()
    lego_lst = []

    calib.read_params_file()

    frame = cv.imread('test2.jpg')

    # Fazer o undistort da imagem, usando o modulo calib
    # undistort_img = calib.undistort(frame)
    undistort_img = fe.resize(frame, constants_feat.RESIZING_FACTOR, constants_feat.RESIZING_FACTOR)

    # Procurar por legos
    fe.find_color_ratio(undistort_img)
    lego_lst.extend(fe.lst_legos)
    # Get unique legos from the list
    A_set = set(lego_lst)
    lego_lst = list(A_set)
    print("Legos found:")
    print(lego_lst)

    cv.imshow('Output Window', undistort_img)

    k = cv.waitKey(-1)

    if k == ord('q'):
        pass

    elif k == ord('s'):
        r = input('What is the name of the output file?\n')
        write2file(r)

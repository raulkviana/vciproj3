import numpy as np
import cv2 as cv
from Constants import constants_app
from calib import Calibration
from lego import Lego
from feature_extrac import FeatureExtrac as fe
from picamera import PiCamera
from time import sleep
from Constants import constants_feat
import csv

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FUNCTIONS
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


def compile2save(lst):
    out_lst = []
    for elem in lst:
        temp = []

        # Append color and ratio
        str1 = elem.color + '/' + str(elem.ratio[0]) + 'x' + str(elem.ratio[1])
        temp.append(str1)

        # Append if it is rect or non rect
        if elem.rect:
            temp.append('rect')
        else:
            temp.append('non-rect')

        # Append contour
        temp.append(elem.contour.tolist()[0])

        # Append to output list
        out_lst.append(temp)
    return out_lst


def write2file(name):
    with open(name + '.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(constants_app.FIELDS_OUTPUT_FILE)
        # Write legos to csv file
        write.writerows(compile2save(lego_lst))


"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MAIN
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

if __name__ == "__main__":
    calib = Calibration()
    fe = fe()
    lego_lst = []

    '''
    SETUP
    '''
    print('Setting up...')


    # Basic setup
    #camera.resolution = constants_app.PICTURES_DIMENSION
    #camera.framerate = constants_app.FPS
    """
    # Configure camera (optional)
    print('Preparing camera color parameters')
    # Force sensor mode 3 (the long exposure mode), set
    # the framerate to 1/6fps, the shutter speed to 6s,
    # and ISO to 800 (for maximum gain)
    camera.sensor_mode = 3
    camera.shutter_speed = 10_000
    camera.iso = 200
    # Give the camera a good long time to set gains and
    # measure AWB (you may wish to use fixed AWB instead)
    sleep(5)
    # Now fix the values
    camera.exposure_mode = 'off'
    g = camera.awb_gains
    camera.awb_mode = 'off'
    camera.awb_gains = g
    sleep(2)

    """

    # Start showing camera
    #camera.start_preview()
    #sleep(5)

    # Calibration module
    r = input('Do you want to calibrate?\n')
    if r == 'yes' or r == 'y':
        # Instrinsic calibration parameters
        #calib.compute_calib_params(piCam=camera)
        calib.compute_calib_params(vid_source=0)

        r = input('Do you want to save parameters to a file?\n')

        if r == 'y':
            calib.write_param_out()

    else:
        print('Reading default parameters...')
        calib.read_params_file()

    r = input('Do you want to get a new reference?\n')

    if r == 'y':
        r = input('Insert the piece unit length (e.g., if it is a 2x2, its length is 2): \n')
        unit_length = int(r)

        print('Position the piece below the camera, when ready press any key')
        input('')

        frame = np.empty((constants_app.PICTURES_DIMENSION[0] * constants_app.PICTURES_DIMENSION[1] * 3,),
                         dtype=np.uint8)
        camera.capture(frame, 'bgr')
        frame = frame.reshape((constants_app.PICTURES_DIMENSION[1], constants_app.PICTURES_DIMENSION[0], 3))

        fe.get_reference(frame, unit_length)

        print('Reference obtained!')

    # App module starting
    print('Starting lego search !!!')
    print('\n\nInstructions:')
    print('Press s to save legos to file')
    print('Press q to quit the app')
    camera = cv.VideoCapture(0)

    while (True):

        ret, frame = camera.read()

        # Pegar numa imagem do picamera e converter para uma imagem utilizavel pelo opencv
        #frame = np.empty((constants_app.PICTURES_DIMENSION[0] * constants_app.PICTURES_DIMENSION[1] * 3,),
                         #dtype=np.uint8)
        #camera.capture(frame, 'bgr')
        #frame = frame.reshape((constants_app.PICTURES_DIMENSION[1], constants_app.PICTURES_DIMENSION[0], 3))

        # undistort_img = calib.undistort(frame)

        # Procurar por legos
        fe.find_color_ratio(frame)
        lego_lst.extend(fe.lst_legos)
        # Get unique legos from the list
        A_set = set(lego_lst)
        # lego_lst = list(A_set)
        print("Legos found:")
        print(lego_lst)

        # Fazer o undistort da imagem, usando o modulo calib
        frame = fe.resize(frame, constants_feat.RESIZING_FACTOR, constants_feat.RESIZING_FACTOR)

        cv.imshow('Output Window', frame)

        k = cv.waitKey(2)
        if k == ord('q'):
            break
        elif k == ord('s'):
            r = input('What is the name of the output file?\n')
            write2file(r)

    camera.stop_preview()

import glob
import cv2 as cv

directory_str = 'dataset/'
scale_percent = 20
windowNameConfig = 'Input'
directory_path = '../dataset_iterator/dataset/'

def resize(img,s=scale_percent):
    width = int(img.shape[1] * s / 100)
    height = int(img.shape[0] * s / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def change_file():
    pass

def mod_pictures(img):
    '''
    Example function

    :param img:
    :return:
    '''
    pass


def window_with_trackbar(wName=windowNameConfig, dir_path=directory_path, mod_pics_funct=mod_pictures, scale_per=20):
    '''
    Shows original image and modifies the original image with a function and shows it

    :param scale_per: Image scaling parameter
    :param wName: Name of the input window
    :param dir_path: directory with the path of dataset
    :param mod_pics_funct: function that modifies the original image
    :return:
    '''

    pics = glob.glob(dir_path + '*.jpg')
    cv.namedWindow(wName)
    cv.namedWindow('Output')
    cv.createTrackbar('File',wName,0,len(pics)-1,change_file)

    while(1):
        picPos = cv.getTrackbarPos('File', wName)
        filename = pics[picPos]

        # Read Image
        img = cv.imread(filename)

        # Resize image
        imgRe = resize(img,scale_per)

        # Show original images
        cv.imshow(windowNameConfig, imgRe)

        newImg = mod_pics_funct(imgRe)

        cv.imshow('Output', newImg)

        if cv.waitKey(1) & 0xff == 27:
            cv.destroyAllWindows()
            break


for file in glob.glob(directory_str + '*.jpg'):
    img = cv.imread(file)
    img_resized = resize(img)
    cv.imshow('resized', img_resized)
    img_gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', img_gray)

    k = cv.waitKey(0)
    if k == ord('q'):
        break

cv.destroyAllWindows()

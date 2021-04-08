import glob
import cv2 as cv

directory_str = 'dataset/'
scale_percent = 20


def resize(img):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


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

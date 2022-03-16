import cv2 as cv
import numpy as np
from joblib import dump, load


def binarize(file):
    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_th = cv.threshold(im_gray, 128, 255, cv.THRESH_OTSU)
    cv.imwrite(f'step.png', im_th)
    im = cv.bitwise_not(im_th)

    return im



def load_object(file):

    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv.threshold(im_gray, 180, 255, cv.THRESH_OTSU)
    im = cv.bitwise_not(im_gray_th_otsu)
    im = cv.resize(im, (10,40))
    cv.imwrite(f'final.png', im)
    arr = []
    for x in im:
        for y in x:
            if y > 0:
                arr.append(y-254)
            else:
                arr.append(y)

    fin = np.array(arr)
    fin = fin.reshape((1, -1))
    print(fin.shape)

    return fin

def recognize(image):
    clf = load(f'clf.fumo')
    pred = clf.predict(image)
    return pred

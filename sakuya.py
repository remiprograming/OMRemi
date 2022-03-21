import cv2 as cv
import numpy as np
from joblib import dump, load
import sklearn.metrics as skm


def binarize(file):
    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_th = cv.threshold(im_gray, 128, 255, cv.THRESH_OTSU)
    cv.imwrite(f'step.png', im_th)


    return im_th



def load_object(im):



    im = cv.resize(im, (10,40))
    arr = []
    for x in im:
        for y in x:
            if y > 0:
                arr.append(y-254)
            else:
                arr.append(y)

    fin = np.array(arr)
    fin = fin.reshape((1, -1))

    return fin

def recognize(image):
    image = load_object(image)
    clf = load(f'clf.fumo')
    pred = clf.predict(image)
    predp = clf.predict_proba(image)

    return pred, predp

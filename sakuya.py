import cv2 as cv
import numpy as np
from joblib import dump, load
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def binarize(file):
    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_th = cv.threshold(im_gray, 128, 255, cv.THRESH_OTSU)
    cv.imwrite(f'step.png', im_th)


    return im_th



def load_object(im):

    blank = np.zeros((64, 64))
    h, w = im.shape
    hh, ww = blank.shape
    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)
    print(im.shape)
    print(blank.shape)
    try:
        blank[yoff:yoff + h, xoff:xoff + w] = im
    except:
        print('error')

    arr = []
    for x in blank:
        for y in x:
            if y > 0:
                arr.append(y-254)
            else:
                arr.append(y)

    fin = np.array(arr)
    fin = fin.reshape(1, 64, 64, 1)

    return fin

def recognize(image):

    t = load_object(image)
    model = tf.keras.models.load_model('model')
    pred = model.predict(t)
    score = tf.nn.softmax(pred[0])
    return pred, score

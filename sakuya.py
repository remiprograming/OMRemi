import cv2
import cv2 as cv
import numpy as np
from joblib import dump, load
import sklearn.metrics as skm
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
dupac = 0
def binarize(file):
    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_th = cv.threshold(im_gray, 128, 255, cv.THRESH_OTSU)
    cv.imwrite(f'step.png', im_th)


    return im_th



def load_object(im):
    global dupac
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

    cv2.imwrite(f'bloac/object{dupac}.png', blank)
    arr = []
    for x in blank:
        for y in x:
            if y > 0:
                arr.append(y-254)
            else:
                arr.append(y)

    fin = np.array(arr)
    fin = fin.reshape((1, 64, 64, 1))

    dupac += 1
    return fin

def recognize(image):

    # image = load_object(image)
    #
    # forest = load('forest.fumo')
    # pred = forest.predict(image)
    # print(pred)
    # if pred == 1:
    #     model = load('clf.fumo')
    #     predi = model.predict(image)
    #     predp = model.predict_proba(image)
    #     return predi, predp
    # else:
    #     return '10', [0]



    t = load_object(image)

    model = tf.keras.models.load_model('model')
    pred = model.predict(t)
    score = tf.nn.softmax(pred[0])
    return pred, score


def find_notes(centroids, staf):
    notes = []
    duppa = 0
    for x in centroids:
        try:
            notes.append(staf.getPitch(x[0]) + f' {duppa}')
        except:
            notes.append(f'{duppa}')
        duppa +=1
    return notes

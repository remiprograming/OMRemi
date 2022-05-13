# import itertools
# import os
# import csv
# import cv2
import os

import cv2
from muscima.io import parse_cropobject_list
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import resize
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
# from sklearn import metrics
# import sakuya as sak
from joblib import dump, load
# import xml.etree.ElementTree as et
# import tensorflow as tf
#
# from tensorflow.keras import datasets, layers, models
# import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot as plt

# Change this to reflect wherever your MUSCIMA++ data lives
CROPOBJECT_DIR = 'muscima/data/cropobjects_manual'

cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
docs = [parse_cropobject_list(f) for f in cropobject_fnames]

def extract_notes_from_doc(cropobjects):
    """Finds all ``(full-notehead, stem)`` pairs that form
    quarter or half notes. Returns two lists of CropObject tuples:
    one for quarter notes, one of half notes.

    :returns: quarter_notes, half_notes
    """
    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if 'notehead' in c.clsname:
            _has_stem = False
            _has_beam_or_flag = False
            stem_obj = None
            jajo = [c]
            for o in c.outlinks:
                cropobj = _cropobj_dict[o]
                if cropobj.clsname in ['stem', 'flag', 'beam']:
                    jajo.append(cropobj)
            notes.append(jajo)

    full_notes = []
    half_notes = []
    quarter_notes = []
    eighth_notes = []
    sixteenth_notes = []

    for arr in notes:
        if arr[0].clsname == 'notehead-empty':
            stem = False
            flagbeams = 0
            for obj in arr:
                if obj.clsname == 'stem':
                    stem = True
                elif obj.clsname.endswith('flag') or obj.clsname == 'beam':
                    flagbeams += 1
            if not stem:
                full_notes.append(arr)
            if stem and flagbeams == 0:
                half_notes.append(arr)
        elif arr[0].clsname == 'notehead-full':
            stem = False
            flagbeams = 0
            for obj in arr:
                if obj.clsname == 'stem':
                    stem = True
                elif obj.clsname.endswith('flag') or obj.clsname == 'beam':
                    flagbeams += 1
            if stem and flagbeams == 0:
                quarter_notes.append(arr)
            elif stem and flagbeams == 1:
                eighth_notes.append(arr)
            elif stem and flagbeams == 2:
                sixteenth_notes.append(arr)

    return full_notes, half_notes, quarter_notes, eighth_notes, sixteenth_notes

full_notes, half_notes, quarter_notes, eighth_notes, sixteenth_notes = [], [], [], [], []
for doc in docs:
    full, half, quarter, eighth, sixteenth = extract_notes_from_doc(doc)
    full_notes.extend(full)
    half_notes.extend(half)
    quarter_notes.extend(quarter)
    eighth_notes.extend(eighth)
    sixteenth_notes.extend(sixteenth)


# print(full_notes)
# print(half_notes)
# print(quarter_notes)
# print(eighth_notes)
# print(sixteenth_notes)

def get_image(cropobjects, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
    There will be a given margin of background on the edges."""

    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])

    # Create the canvas onto which the masks will be pasted
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    canvas = numpy.zeros((height, width), dtype='uint8')

    for c in cropobjects:
        # Get coordinates of upper left corner of the CropObject
        # relative to the canvas
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        # We have to add the mask, so as not to overwrite
        # previous nonzeros when symbol bounding boxes overlap.
        canvas[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask

    canvas[canvas > 0] = 1
    return canvas


full_images = [get_image(n) for n in full_notes]
half_images = [get_image(n) for n in half_notes]
quarter_images = [get_image(n) for n in quarter_notes]
eighth_images = [get_image(n) for n in eighth_notes]
sixteenth_images = [get_image(n) for n in sixteenth_notes]


def show_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.show()

def show_masks(masks, row_length=5):
    n_masks = len(masks)
    n_rows = n_masks // row_length + 1
    n_cols = min(n_masks, row_length)
    fig = plt.figure()
    for i, mask in enumerate(masks):
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(mask, cmap='gray', interpolation='nearest')
    # Let's remove the axis labels, they clutter the image.
    for ax in fig.axes:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
    plt.show()

# print("quarter", quarter_images)
# print("eifht", eighth_images)
# print("sixteen", sixteenth_images)
show_masks(full_images[:25])
show_masks(half_images[:25])
show_masks(quarter_images[:25])
show_masks(eighth_images[:25])
show_masks(sixteenth_images[:25])

def paste(im):
    #resize image im to be of size (64, 64) preserving aspect ratio
    scale_factor = 64 / max(im.shape[1], im.shape[0])
    im_resized = cv2.resize(im, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    blank = numpy.zeros((64, 64))
    h, w = im_resized.shape
    hh, ww = blank.shape
    yoff = round((hh - h) / 2)
    xoff = round((ww - w) / 2)
    try:
        blank[yoff:yoff + h, xoff:xoff + w] = im_resized
    except:
        print('error')

    return blank


from skimage.transform import resize

full_resized = [paste(n) for n in full_images]
half_resized = [paste(n) for n in half_images]
quarter_resized = [paste(n) for n in quarter_images]
eighth_resized = [paste(n) for n in eighth_images]
sixteenth_resized = [paste(n) for n in sixteenth_images]


# And re-binarize, to compensate for interpolation effects
full_resized = [numpy.where(n > 0, 1, 0) for n in full_resized]
half_resized = [numpy.where(n > 0, 1, 0) for n in half_resized]
quarter_resized = [numpy.where(n > 0, 1, 0) for n in quarter_resized]
eighth_resized = [numpy.where(n > 0, 1, 0) for n in eighth_resized]
sixteenth_resized = [numpy.where(n > 0, 1, 0) for n in sixteenth_resized]


show_masks(full_resized[:25])
show_masks(half_resized[:25])
show_masks(quarter_resized[:25])
show_masks(eighth_resized[:25])
show_masks(sixteenth_resized[:25])

F_LABEL = 0
H_LABEL = 1
Q_LABEL = 2
E_LABEL = 3
S_LABEL = 4

full_labels = [F_LABEL for _ in full_resized]
half_labels = [H_LABEL for _ in half_resized]
quarter_labels = [Q_LABEL for _ in quarter_resized]
eighth_labels = [E_LABEL for _ in eighth_resized]
sixteenth_labels = [S_LABEL for _ in sixteenth_resized]

print(len(full_resized))
print(len(half_resized))
print(len(quarter_resized))
print(len(eighth_resized))
print(len(sixteenth_resized))

notes = full_resized + half_resized + quarter_resized + eighth_resized + sixteenth_resized
labels = full_labels + half_labels + quarter_labels + eighth_labels + sixteenth_labels
notes_flat = [n.flatten() for n in notes]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    notes_flat, labels, test_size=0.25, random_state=42,
    stratify=labels)

from sklearn.neighbors import KNeighborsClassifier

K=5

clf = KNeighborsClassifier(n_neighbors=K)
clf.fit(X_train, y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
y_test_pred = clf.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred, target_names=['F', 'H', 'Q', 'E', 'S']))
dump(clf, 'clf.fumo')

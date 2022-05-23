# import itertools
# import os
# import csv
# import cv2
import os
from itertools import cycle

import cv2
import numpy as np
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

classes = []

def extract_notes_from_doc(cropobjects):
    """Finds all ``(full-notehead, stem)`` pairs that form
    quarter or half notes. Returns two lists of CropObject tuples:
    one for quarter notes, one of half notes.

    :returns: quarter_notes, half_notes
    """
    global classes

    _cropobj_dict = {c.objid: c for c in cropobjects}

    notes = []
    for c in cropobjects:
        if not c.clsname in classes:
            classes.append(c.clsname)
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
    bloat = []

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
        else:
            bloat.append(arr)

    return full_notes, half_notes, quarter_notes, eighth_notes, sixteenth_notes, bloat

full_notes, half_notes, quarter_notes, eighth_notes, sixteenth_notes, bloat = [], [], [], [], [], []
for doc in docs:
    full, half, quarter, eighth, sixteenth, blt = extract_notes_from_doc(doc)
    full_notes.extend(full)
    half_notes.extend(half)
    quarter_notes.extend(quarter)
    eighth_notes.extend(eighth)
    sixteenth_notes.extend(sixteenth)
    bloat.extend(blt)

# print(classes)
# quit()

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
bloat_images = [get_image(n) for n in bloat]

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
# show_masks(full_images[:25])
# show_masks(half_images[:25])
# show_masks(quarter_images[:25])
# show_masks(eighth_images[:25])
# show_masks(sixteenth_images[:25])
# show_masks(bloat_images[:25])

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
    print(blank.shape)
    return blank


from skimage.transform import resize

full_resized = [paste(n) for n in full_images]
half_resized = [paste(n) for n in half_images]
quarter_resized = [paste(n) for n in quarter_images]
eighth_resized = [paste(n) for n in eighth_images]
sixteenth_resized = [paste(n) for n in sixteenth_images]
bloat_resized = [paste(n) for n in bloat_images]


# And re-binarize, to compensate for interpolation effects
full_resized = [numpy.where(n > 0, 1, 0) for n in full_resized]
half_resized = [numpy.where(n > 0, 1, 0) for n in half_resized]
quarter_resized = [numpy.where(n > 0, 1, 0) for n in quarter_resized]
eighth_resized = [numpy.where(n > 0, 1, 0) for n in eighth_resized]
sixteenth_resized = [numpy.where(n > 0, 1, 0) for n in sixteenth_resized]
bloat_resized = [numpy.where(n > 0, 1, 0) for n in bloat_resized]


# show_masks(full_resized[:25])
# show_masks(half_resized[:25])
# show_masks(quarter_resized[:25])
# show_masks(eighth_resized[:25])
# show_masks(sixteenth_resized[:25])
# show_masks(bloat_resized[:25])

F_LABEL = 0
H_LABEL = 1
Q_LABEL = 2
E_LABEL = 3
S_LABEL = 4
B_LABEL = 5

full_labels = [F_LABEL for _ in full_resized]
half_labels = [H_LABEL for _ in half_resized]
quarter_labels = [Q_LABEL for _ in quarter_resized]
eighth_labels = [E_LABEL for _ in eighth_resized]
sixteenth_labels = [S_LABEL for _ in sixteenth_resized]
bloat_labels = [B_LABEL for _ in bloat_resized]

print(len(full_resized))
print(len(half_resized))
print(len(quarter_resized))
print(len(eighth_resized))
print(len(sixteenth_resized))
print(len(bloat_resized))

notes = full_resized + half_resized + quarter_resized + eighth_resized + sixteenth_resized
labels = full_labels + half_labels + quarter_labels + eighth_labels + sixteenth_labels
notes_flat = [n.flatten() for n in notes]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(notes_flat, labels, test_size=0.25, random_state=42,stratify=labels)
X_train = numpy.array(X_train).reshape((-1, 64, 64, 1))
X_test = numpy.array(X_test).reshape((-1, 64, 64, 1))
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)



# from sklearn.neighbors import KNeighborsClassifier
#
# K=5


# arr = []
# l=len(X_train)
# i=1
# for sample in X_train:
#     print(f'{i}/{l}')
#     sample = sample.tolist()
#     arr.extend(sample)
#     i+=1
#
# arr = numpy.array(arr)
# arr = arr.reshape((64**2, len(X_train)))
# print(arr.shape)
# print(arr)
# from sklearn.metrics import classification_report, roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
#
# from sklearn.ensemble import IsolationForest
# forest = IsolationForest(n_estimators=100, max_samples=128, contamination=(len(bloat_resized)/len(X_train)))
# forest.fit(X_train)
#
#
# predictions = forest.predict(X_test)
# print(classification_report(y_test, predictions))
# cm = confusion_matrix(y_test, predictions)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()

# dump(forest, 'forest.fumo')

# clf = KNeighborsClassifier(n_neighbors=K)
# clf.fit(X_train, y_train)
#
# y_test_pred = clf.predict(X_test)


# print(classification_report(y_test, y_test_pred, target_names=['F', 'H', 'Q', 'E', 'S', 'B']))
# cm = confusion_matrix(y_test, y_test_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
# disp.plot()
# plt.show()
# # Compute ROC curve and ROC area for each class
# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# # for i in range(6):
# #     fpr[i], tpr[i], _ = roc_curve(y_test, y_test_pred)
# #     roc_auc[i] = auc(fpr[i], tpr[i])
# #
# #
# # for i in range(6):
# #     plt.plot(fpr[i],tpr[i],label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))
# # plt.show()
#
# dump(clf, 'clf.fumo')
import tensorflow as tf
from tensorflow.keras import layers, models


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(test_acc)

model.save('model')
import itertools
import os
import csv
import cv2
import muscima
from muscima import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
import sakuya as sak
from joblib import dump, load
import xml.etree.ElementTree as et
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

t = []
cs = csv.reader(open('train/data1.csv', 'r'))
for x in cs:
    if len(x) > 0:
        t.append(x)

t.pop(0)

labels = []
for i in range(len(t)):
    labels.append(t[i][0])


note = []
for x in t:
    x.pop(0)
    x = [int(i) for i in x]
    for i in range(len(x)):
        if x[i] > 128:
            x[i] = 0
        else:
            x[i] = 1

    y = np.array(x)
    y = y.reshape(64, 64)
    note.append(y)



data = list(zip(note, labels))

random.shuffle(data)

note, labels = zip(*data)

print(len(note), len(labels))

label = []
for x in labels:
    if x == 'Whole':
        label.append(0)
    elif x == 'Half':
        label.append(1)
    elif x == 'Quarter':
        label.append(2)
    elif x == 'Eight':
        label.append(3)
    elif x == 'Sixteenth':
        label.append(4)



test_images = np.array(note[0:25])
test_labels = np.array(label[0:25])


note = np.array((note[25:]))
labels = np.array((label[25:]))

print(len(note), len(labels))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])




history = model.fit(note, labels, epochs=5, validation_data=(test_images, test_labels))

model.save('model')




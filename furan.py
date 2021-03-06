import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import skimage.measure

import lyrica

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import cv2

import sakuya, staff

def flanderize(image):
    image = cv2.bitwise_not(image)
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)


    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    boxes = []

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 10:

            if region.extent <= 0.85:
                boxes.append(region)
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)

    #sort by x

    boxes.sort(key=lambda x: x.centroid[1])


    plt.savefig(f'final.png')
    return boxes

def remilialize(file):
    image = sakuya.binarize(file)
    x = staff.getStaffData(image)

    im = staff.removeStaffLinesSP(image, x)

    regions = flanderize(im)

    im = cv2.bitwise_not(im)
    imarr=[]
    centroids = []
    i = 0
    for reg in regions:
        mr, mc, xr, xc = reg.bbox
        temp = im[mr:xr, mc:xc]
        imarr.append(temp)
        centro = list(reg.centroid)
        cent = list(skimage.measure.regionprops(temp, temp)[0].centroid_weighted)
        cent[0] += centro[0]
        cent[1] += centro[1]

        centroids.append(cent)

        cv2.imwrite(f'bloat/object{i}.png', temp)
        i += 1

    score = []
    preds=[]
    i = 0
    for im in imarr:
        pr, scr = sakuya.recognize(im)
        preds.append(f'{i}. {pr}')
        score.append(np.argmax(scr))
        i+=1


    print(score)
    lyrica.build_midi(sakuya.find_notes(centroids, x), score)





file = f'hard.png'
remilialize(file)








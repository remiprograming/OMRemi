import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
            boxes.append(region)
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)



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

        centroids.append(reg.centroid)
        imarr.append(temp)
        cv2.imwrite(f'bloat/object{i}.png', temp)
        i += 1

    score = []
    preds=[]
    i = 0
    for im in imarr:
        pr, scr = sakuya.recognize(im)
        preds.append(pr)
        score.append(f'{i}. {np.argmax(scr)}')
        i+=1

    print(score)

    print(sakuya.find_notes(centroids, x))


    conf = -1
    detected = []
    i = 0
    for ped in preds:
        for x in ped[1]:
            y = x.tolist()
            for h in y:
                if h > conf:
                    print(ped[0], h)
                    detected.append((ped[0], regions[i]))
        i+=1






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

    for region in detected:
        # take regions with large enough areas
        if region[1].area >= 10:
            boxes.append(region[1])
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region[1].bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            cv2.putText(img=image, text=str(region[0][0]), org=(minc, maxr), fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3,
                        color=(255, 255, 255), thickness=1)

            ax.add_patch(rect)
    cv2.imwrite(f'final3.png', image)


    plt.savefig(f'final2.png')




file = f'score_0.png'
remilialize(file)








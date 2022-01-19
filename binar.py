import cv2 as cv

def bin(file):

    im = cv.imread(file)

    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    th, im_gray_th_otsu = cv.threshold(im_gray, 128, 255, cv.THRESH_OTSU)

    cv.imwrite('final.png', im_gray_th_otsu)

bin(f'im.png')
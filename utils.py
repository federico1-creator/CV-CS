print('file con le varie funzioni')
import numpy as np
import cv2

def pres():
    print('funzioni')

# NOT USED
def water(image, thresh):
    print('passo ad utils')

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('opening', opening)
    cv2.waitKey(0)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imshow('sure_bg', sure_bg)
    cv2.waitKey(0)

    # Finding sure foreground area
    opening[opening == 255] = 1
    opening = np.asarray(opening, dtype=np.uint8)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # unknown= sure_bg
    cv2.imshow('opening', unknown)
    cv2.waitKey(0)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    print('mak:', markers.max(), markers.min(), markers.shape)
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    print('mark:', markers.max(), markers.min(), markers.shape)
    image[markers == -1] = [255, 0, 0]
    cv2.imshow('final', image)
    cv2.waitKey(0)


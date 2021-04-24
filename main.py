# file iniziale
import torch
import cv2 # BGR order
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *

def showRes(H, W, image):
    resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imshow('image resized', resized)
    cv2.waitKey(0)
    # imS = cv2.resize(image, (960, 540))

def openimage():# aprire l'immagine dataset di dimensioni 1024,768
    image= cv2.imread('data-prova\\12141279ui_0_r.jpg') # 0
    # img= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pres()
    cv2.imshow('imported image', image)
    cv2.waitKey(0)
    '''
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    # resize operation
    height = int(image.shape[0] * 20 / 100)
    width = int(image.shape[1] * 20 / 100)
    showRes(height, width, image)

    cv2.imwrite('output/copy.png',image)
    return image, height, width

def sogliaRGB(image):
    # try & except per vedere se image ha la 3° dimensione
    try:
        image.shape[2]
    except:
        print('Immagine non a colori')
    else:
        low = (0, 127, 0) # normale threshold
        high = (127, 255, 255)
        mask = cv2.inRange(image, low, high)

        cv2.imshow('threshold in RGB', mask)
        cv2.waitKey(0)

        th = np.zeros((image.shape[0], image.shape[1], image.shape[2])) # adaptive threshold (1024, 768, 3)
        for i in range(image.shape[2]):
            th[:, :, i] = cv2.adaptiveThreshold(image[:, :, i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11,
                                                    0)
        cv2.imshow('adaptive threshold in RGB', th)
        cv2.waitKey(0)

def soglia(image):
    # 0
    _, thresh0 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # thresh= cv2.resize(thresh, (592,592))
    cv2.imshow('static threshold', thresh0)
    cv2.waitKey(0)

    try:
        image.shape[2]
    except:
        print('Immagine non a colori')

        # 1
        ret1, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        print('Soglia usata:', ret1)
        cv2.imshow('OTSUs threshold', thresh1)
        cv2.waitKey(0)
        # 2
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret2, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('OTSU + blur', thresh2)
        cv2.waitKey(0)
        # 3
        thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0)

        cv2.imshow('adaptive threshold', thresh3)
        cv2.waitKey(0)

def canny(image):
    image= cv2.Canny(image, 100, 200)
    cv2.imshow('image with Canny', image)
    cv2.waitKey(0)

def sobel(image):
    kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])

    sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    cv2.imshow('sobel hor', sobel_horizontal)
    cv2.waitKey(0)
    cv2.imshow('sobel ver', sobel_vertical)
    cv2.waitKey(0)
    # together

if __name__ == "__main__":
    print('Inizio progetto')
    image, H, W= openimage() # immagine aperta nell'oggetto image
    print(image.shape)

    # SOGLIE
    soglia(image) # thresh: immagine dove applicata la soglia
    sogliaRGB(image)

    # SOBEL
    sobel(image)
    # CANNY
    canny(image)
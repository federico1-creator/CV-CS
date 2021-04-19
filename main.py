# file iniziale
import torch
import cv2
import matplotlib.pyplot as plt
import os
from utils import *

def show(H, W, image):
    resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    cv2.imshow('image resized', resized)
    cv2.waitKey(0)
    # imS = cv2.resize(image, (960, 540))
    
def openimage():# aprire l'immagine dataset di dimensioni 1024,768
    image= cv2.imread('data-prova\\12141279ui_0_r.jpg', 0) # 0
    print(image.shape)
    pres()

    cv2.imshow('image', image)
    cv2.waitKey(0)
    '''
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    # resize operation
    height = int(image.shape[0] * 20 / 100)
    width = int(image.shape[1] * 20 / 100)
    show(height, width, image)

    cv2.imwrite('output/copy.png',image)
    return image, height, width

def soglia1(image):
    # 1° ritorno è la soglia stabilita
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY) # TODO
    return thresh
    #thresh= cv2.resize(thresh, (592,592))

def soglia2(image):

    pass


if __name__ == "__main__":
    print('Inizio progetto')

    image, H, W= openimage()
    thresh= soglia1(image)
    show(H, W, thresh)
    soglia2(image)

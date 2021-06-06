'''
Verificare i metodi in tutte le immagini, per avlutare ed estendere la loro efficacia.
Lavoro sul dataset UPPER_BODY
'''
import torch
import numpy as np
import cv2
import time
import os

PATH_easy= 'data/easy-segm'
PATH_hard= 'data/hard-segm'

def kmeans_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change color to RGB (from BGR)
    pixel_vals = image.reshape((-1, 3))  # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = np.float32(pixel_vals)  # Convert to float type only for supporting cv2.kmean
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)  # criteria
    k = 3  ### Choosing number of cluster
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # LABELS: ha la classe per ogni pixel, con dimensione spiattellata
    centers = np.uint8(centers)  # convert data into 8-bit values
    segmented_data = centers[labels.flatten()]  # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape))  # reshape data into the original image dimensions

    return segmented_image

def loop_images(PATH= PATH_easy):
    for image in os.listdir(PATH):
        img= cv2.imread((os.path.join(PATH, image)))

        # funzione da applicare
        img_segm= kmeans_color(img)

        cv2.imshow('loop delle immagini', img_segm)
        cv2.waitKey(0)

if __name__ == '__main__':
    loop_images(PATH=PATH_hard)

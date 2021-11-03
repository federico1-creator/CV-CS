print('file con le varie funzioni')
import numpy as np
import cv2
import torch

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

# ---
def drawing_bbox(image, mask):
    print('disegno del bbox')

    mask = cv2.Canny(mask, threshold1=100, threshold2=200)
    coun, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # with dtype=uint8
    pos = []  # posizione del vestito nell'immagine
    for c in coun:
        x, y, w, h = cv2.boundingRect(c)
        if w > 120 and h > 120:  # per filtrare i bordi      # w > 170
            dress1 = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            pos.append([x, y, w, h])

    # restituisco i pixel dove si trova la maglietta
    cv2.imshow('bbox with Canny', dress1)
    cv2.waitKey(0)
    return pos # return x, y, w, h

def extract_mask(mask, image):
    # quando devo estrarre dalla maschera solo una classe che mi interessa
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    mask= torch.as_tensor(mask)
    uniq=mask[:,:,:].unique()
    print(mask.shape)
    print(uniq)

    h, w, c = mask.shape
    restructured= torch.zeros(h,w,c)
    val=[128, 0, 0] # rappresenta il colore blu, problema con i colori che hanno una parte di blu (128 nel primo canale)
    if mask.shape[2]==3:
        stesa0 = mask[:, :, 0].reshape(-1,1)
        stesa1 = mask[:, :, 1].reshape(-1, 1)
        stesa2 = mask[:, :, 2].reshape(-1, 1)

    for j in range(len(stesa0)): # da mettere con condizione (and) per vedere se altri 2 canali con valori del colore che voglio
        if (stesa0[j] == val[0]) and (stesa1[j] == val[1]) and (stesa2[j] == val[2]):  # da selezionare il codice del colore RGB esatto
            stesa0[j] = 128
            stesa1[j] = 0
            stesa2[j] = 0
        else:
            stesa0[j] = 0
            stesa1[j] = 0
            stesa2[j] = 0

    stesa0= stesa0.reshape(h,w)
    stesa1 = stesa1.reshape(h, w)
    stesa2 = stesa2.reshape(h, w)
    restructured= torch.stack((stesa0,stesa1,stesa2), dim=2)
    print(restructured.shape) # controllare sotto
    a,b,z= restructured[:,:,0].unique(return_counts=True, return_inverse=True)

    blocco= restructured[restructured[:,:,0]==128]
    restructured = np.array(restructured)
    cv2.imshow('res', restructured)
    cv2.waitKey(0)
    # una volta che ho solo un colore posso fare bbox, con canny e bordi per disegnare il rettangolo

    # ritorna i 4 valori del bbox
    bbox_pos= drawing_bbox(image, restructured)
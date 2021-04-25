# file iniziale
#TODO: HOG;
#da completare: template matching, contour (con il concetto di hierarchy)
import torch
import cv2 # BGR order
import numpy as np
import matplotlib.pyplot as plt
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

def colors():
    c= np.zeros(10)
    c[0]= 0
    c[1]= 100
    c[2]= 50
    c[3]= 100
    c[4]= 100
    c[5]= 150
    c[6]= 180
    c[7]= 200
    c[8]= 220
    c[9]= 255
    return c

def resize(image, scale):
    # resize operation
    height = int(image.shape[0] * scale // 10)
    width = int(image.shape[1] * scale // 10)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('image resized', resized)
    cv2.waitKey(0)
    print(resized.shape)
    return resized

def hist(im, nbin):
    print(im.shape)
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist(im, [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([-10, 260])

    plt.show() # 256 diversi bin

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
    image= cv2.Canny(image, 100, 200) # 180, 600
    cv2.imshow('image with Canny', image)
    cv2.waitKey(0)

def sobel(image):
    # kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])

    sobel_horizontal = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_vertical = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    cv2.imshow('sobel hor', sobel_horizontal)
    cv2.waitKey(0)
    cv2.imshow('sobel ver', sobel_vertical)
    cv2.waitKey(0)
    # together

def edgeHSV(im):
    im= cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    im = torch.from_numpy(im)
    H, W = im.shape[0], im.shape[1]
    kH, kW = 3, 3
    kernel = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]])  # , dtype=int

    oH = H - (kH - 1)
    oW = W - (kW - 1)
    edge = torch.zeros((2, oH, oW))  # oC

    for i in range(oH):
        for j in range(oW):
            edge[:, i, j] = (im[i:i + kH, j:j + kH] * kernel).sum(axis=(1, 2))
    print(edge.shape)

    magnitude = torch.sqrt(torch.pow(edge[0], 2) + torch.pow(edge[1], 2))
    teta = np.abs(np.arctan2(edge[1], edge[0]))  # abs,       rimane un tensore
    # normalization
    magnitude = (magnitude - magnitude.min()) / (1081 - magnitude.min())  # min=0  /1081
    teta = (teta * 180) / np.pi  # tra 0 e 180
    # HSV space
    HSV0 = teta
    HSV1 = (torch.zeros((1022, 766))) + 255
    HSV2 = magnitude
    HSV = torch.stack((HSV0, HSV1, HSV2), axis=-1)  # []
    # to RGB space
    final = cv2.cvtColor(np.float32(HSV), cv2.COLOR_HSV2RGB)
    plt.imshow(final)  # (126, 126, 3)
    plt.show()

def contour(image):
    image_copy = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # faccio prove con contorni
    image= cv2.Canny(image, threshold1=100, threshold2=200) # immagine diventa ad unico canale, lavorare sui parametri
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    #RETR_LIST se non uso la gerarchia, # RETR_
    contours, hierarchy = cv2.findContours(image=image,
                                           mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # con _NONE valore di 600

    hierarchy= np.squeeze(hierarchy)
    print(hierarchy)
    #j=0
    c = []
    for i in range(len(contours)):
        if contours[i].shape[0] >= 200: # deve avere dimensione precisa dei valori utulizzati
            c.append(contours[i])
            #j=j+1
    '''
    h = np.zeros((j+1, 4))
    for i in range(len(contours)):
        if contours[i].shape[0] >= 200:
            h[i]= hierarchy[i]
    '''
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, # contours=c
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('vediamo', image_copy)
    cv2.waitKey(0)

def template(image1, image2, col): # delle stesse dimensioni (1024, 768, 3)
    #image1= cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    #image2= cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    for i in [2, 4, 6, 8]:
        temp = image2
        temp= resize(temp, i) # chiamo la funzione

        w, h = temp.shape[0], temp.shape[1]
        method= cv2.TM_CCORR_NORMED
        # Apply template Matching
        res = cv2.matchTemplate(image1,temp,method) # mappa in 2D
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image1, top_left, bottom_right, col[i], 2) # -1 per riempire il rettangolo
        # plot
        plt.subplot(221),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(temp, cmap='gray')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(image1,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()


if __name__ == "__main__":
    print('Inizio progetto')
    image, H, W= openimage() # immagine aperta nell'oggetto image
    image2 = cv2.imread('data-prova\\12141279ui_1_f.jpg')
    print(image.shape)
    # HIST
    hist(image, 10)
    # SOGLIE
    soglia(image) # thresh: immagine dove applicata la soglia
    sogliaRGB(image)
    # SOBEL
    sobel(image)
    # CANNY
    canny(image)
    # edge with colors
    edgeHSV(image)
    # CONTOURS
    contour(image)
    # TEMPLATE MATCHING
    col = colors()
    template(image, image2, col)  # resize



''' parametri Canny
    image_gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    m = np.median(image_gray)
    sigma= 0.33
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
'''
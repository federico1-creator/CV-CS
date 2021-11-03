'''file iniziale

HOG per il nostro caso può essere utile?
'''
# for image in os.listdir(''):
#      image = cv2.imread(os.path.join('path_to_images_folder', image))
import time
import os
import torch
import cv2  # BGR order
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import skimage.segmentation


def dimension(mask):
    cv2.imshow('mask', mask)
    cv2.waitKey(0)

    mask= torch.as_tensor(mask)
    uniq=mask[:,:,:].unique()
    print(mask.shape)
    print(uniq)

    h, w, c = mask.shape
    restructured= torch.zeros(h,w,c)
    val=[128, 0, 0] # rappresenta il colore blu, problema con i colori che hanno una parte di blu (128 nel primo canale)
    for i in range(c):
        work_on = mask[:,:,i]
        stesa = work_on.reshape(-1, 1)
        print('val unici', stesa.unique())
        for j in range(len(stesa)): # da mettere con condizione (and) per vedere se altri 2 canali con valori del colore che voglio
            if stesa[j] != val[i]:  # da selezionare il codice del colore RGB esatto
                stesa[j] = 0
        restructured[:,:,i]= stesa.reshape(h,w)
    print(restructured.shape) # controllare sotto
    a,b= restructured[:,:,0].unique(return_counts=True)

    restructured= restructured.reshape(h,w,c)
    restructured = np.array(restructured)
    print(restructured.shape)
    cv2.imshow('res', restructured)
    cv2.waitKey(0)
    input()

    h,w,c= mask.shape
    stesa= mask.reshape(-1,1)
    for i in range(len(stesa)):
        if stesa[i] != 128: # da selezionare il codice del colore RGB esatto
            stesa[i]=0
    print(stesa.unique())
    structured= stesa.reshape(h,w,c)
    structured= np.array(structured)
    print(structured.shape)
    cv2.imshow('mask', structured)
    cv2.waitKey(0)

def showRes(H, W, image):
    resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    #cv2.imshow('image resized', resized)
    #cv2.waitKey(0)

    # imS = cv2.resize(image, (960, 540))


def openimage(path):  # aprire l'immagine dataset di dimensioni 1024,768
    image = cv2.imread(path)  # 0
    img= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    pres()
    cv2.imshow('imported image', image)
    cv2.waitKey(0)
    '''
    plt.imshow(image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    '''
    # resize operation NOT USED
    height = int(image.shape[0] * 20 / 100)
    width = int(image.shape[1] * 20 / 100)
    showRes(height, width, image)

    cv2.imwrite('output/copy.png', image)
    return image, height, width, img


def colors():
    c = np.zeros(10)
    c[0] = 0
    c[1] = 100
    c[2] = 50
    c[3] = 100
    c[4] = 100
    c[5] = 150
    c[6] = 180
    c[7] = 200
    c[8] = 220
    c[9] = 255
    return c

# util in the template matching
def resize(image, scale):
    # resize operation
    height = int(image.shape[0] * scale // 10)
    width = int(image.shape[1] * scale // 10)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.imshow('image resized', resized)
    cv2.waitKey(0)
    print(resized.shape)
    return resized

def morp_closing(image, dimker=5):
    # remove holes
    kernel = np.ones((dimker, dimker), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imshow('closing', closing)
    cv2.waitKey(0)
    return closing

def morp_dilate(image, dimker=5):
    # sure background area
    kernel = np.ones((dimker, dimker), np.uint8)
    sure_bg = cv2.dilate(image, kernel, iterations=3)
    cv2.imshow('dilate', sure_bg)
    cv2.waitKey(0)
    return sure_bg

def morp_OpeningErosion(image):
    # Opening: rimuove il rumore nell'immagine
    # Erosion: immagine più sottile
    pass

def hist(im, nbin):
    print(im.shape)
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist(im, [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([-10, 260])

    plt.show()  # 256 diversi bin


def sogliaRGB(image):
    # try & except per vedere se image ha la 3° dimensione
    try:
        image.shape[2]
    except:
        print('Immagine non a colori')
    else:
        low = (0, 127, 0)  # normale threshold
        high = (127, 255, 255)
        mask = cv2.inRange(image, low, high)

        cv2.imshow('threshold in RGB', mask)
        cv2.waitKey(0)

        th = np.zeros((image.shape[0], image.shape[1], image.shape[2]))  # adaptive threshold (1024, 768, 3)
        for i in range(image.shape[2]):
            th[:, :, i] = cv2.adaptiveThreshold(image[:, :, i], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
                                                11,
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
        thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0) # blockSize=50

        cv2.imshow('adaptive threshold', thresh3)
        cv2.waitKey(0)


def canny(image):
    image = cv2.Canny(image, 100, 200)  # 180, 600
    cv2.imshow('image with Canny', image)
    cv2.waitKey(0)

    return image

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
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
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
    image_copy = image.copy() # BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # gray scale

    # creo contorni
    image= cv2.Canny(image, threshold1=100, threshold2=200) # immagine diventa ad unico canale, lavorare sui parametri
    #_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    #RETR_LIST se non uso la gerarchia, # RETR_
    contours, hierarchy = cv2.findContours(image=image,
                                           mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE) # con _NONE valore di 600

    print(len(contours))  # list
    print(contours[88].shape)  # il primo valore indica il numero di punti del contorno
    # devo avere una FORMA CHIUSA PER DISEGNARE IL RETTANGOLO
    '''
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    '''
    hierarchy= np.squeeze(hierarchy)
    # TODO: capire come poter usare la gerarchia , che viene stampata sotto
    # print(hierarchy)
    j=0
    c = []
    for i in range(len(contours)):
        if contours[i].shape[0] >= 200: # deve avere dimensione precisa dei valori utulizzati
            c.append(contours[i])
            j=j+1
    '''
    h = np.zeros((j+1, 4))
    for i in range(len(contours)):
        if contours[i].shape[0] >= 200:
            h[i]= hierarchy[i]
    '''
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('contorni', image_copy) # sull'immagine a colori
    cv2.waitKey(0)


def template(image1, image2, col):  # delle stesse dimensioni (1024, 768, 3)
    # image1= cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    # image2= cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    for i in [6, 8]:
        temp = image2
        temp = resize(temp, i)  # chiamo la funzione

        w, h = temp.shape[0], temp.shape[1]
        method = cv2.TM_CCORR_NORMED
        # Apply template Matching
        res = cv2.matchTemplate(image1, temp, method)  # mappa in 2D
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(image1, top_left, bottom_right, col[i], 2)  # -1 per riempire il rettangolo
        # plot
        plt.subplot(221), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(temp, cmap='gray')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(image1, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()

        plt.subplot(121), plt.imshow(temp, cmap='gray')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(image1, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.show()


def onTrackbarChange(max_slider):
    cimg = np.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    p1 = max_slider
    p2 = max_slider * 0.4

    # Detect circles using HoughCircles transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, cimg.shape[0] / 64, param1=p1, param2=p2, minRadius=25,
                               maxRadius=50)

    # If at least 1 circle is detected
    if circles is not None:
        cir_len = circles.shape[1]  # store length of circles found
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        cir_len = 0  # no circles detected

    # Display output image
    cv2.imshow('Image', cimg)

    # Edge image for debugging
    edges = cv2.Canny(gray, p1, p2)
    cv2.imshow('Edges', edges)


def onTrackbarChange(max_slider):
    dst = np.copy(image)

    th1 = max_slider
    th2 = th1 * 0.4
    edges = cv2.Canny(image, th1, th2)

    # Apply probabilistic hough line transform
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180.0, 50, minLineLength=10, maxLineGap=100)

    # Draw lines on the detected points
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("Result Image", dst)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


def hog(image):
    hog = cv2.HOGDescriptor()
    # change HOGDescriptor_getDefaultPeopleDetector with our parameters
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, _w = hog.detectMultiScale(image, winStride=(8, 8), padding=(32, 32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(image, found)
    draw_detections(image, found_filtered, 3)
    print('%d (%d) found' % (len(found_filtered), len(found)))
    cv2.imshow('HOG', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def noise(image):
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    blurring= cv2.medianBlur(image,5)
    bilateral= cv2.bilateralFilter(image,9,75,75)

    plt.subplot(221), plt.imshow(image), plt.title('Original image')
    plt.subplot(222), plt.imshow(gaussian), plt.title('Gaussian filter')
    plt.subplot(223), plt.imshow(blurring), plt.title('Blurred filter')
    plt.subplot(224), plt.imshow(bilateral), plt.title('Bilateral filter')
    plt.show()

    return gaussian


def kmeans(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Change color to RGB (from BGR)
    pixel_vals = image.reshape((-1, 3)) # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = np.float32(pixel_vals) # Convert to float type only for supporting cv2.kmean
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)  # criteria
    k = 4  ### Choosing number of cluster
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # LABELS: ha la classe per ogni pixel, con dimensione spiattellata
    centers = np.uint8(centers)  # convert data into 8-bit values
    segmented_data = centers[labels.flatten()]  # Mapping labels to center points( RGB Value)
    segmented_image = segmented_data.reshape((image.shape))  # reshape data into the original image dimensions
    plt.imshow(segmented_image)
    plt.title('k=4')
    plt.show()

def dress(image2):
    # trovato bordi e contorni, da mettere box intorno all'elemento, fatto in 2 modi.

    # immagine a colori with CANNY
    image_copy= image2.copy() # per avere i bbox non sovrapposti sulla stessa immagine
    image = cv2.Canny(image2, threshold1=100, threshold2=200)
    cv2.imshow('Canny', image)
    cv2.waitKey(0)

    coun, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # with dtype=uint8
    pos1 = []  # posizione del vestito nell'immagine
    for c in coun:
        x, y, w, h = cv2.boundingRect(c)
        if w > 120 and h > 120:  # per filtrare i bordi      # w > 170
            dress1 = cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
            pos1.append([x, y, w, h])
    cv2.imshow('bbox with Canny', dress1)
    cv2.waitKey(0)

    # gray image with OTSU
    image_gray= cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    ret1, thresh1 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('Soglia usata:', ret1)
    cv2.imshow('OTSU', thresh1)
    cv2.waitKey(0)

    coun, _ = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pos2=[] # posizione del vestito nell'immagine
    for c in coun:
        x, y, w, h = cv2.boundingRect(c)
        if (w > 120 and h > 120) and (x!=0 and y!=0) : # per filtrare i bordi
            dress2 = cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)
            pos2.append([x, y, w, h])
    cv2.imshow('bbox with OTSU', dress2)
    cv2.waitKey(0)
    return pos1

def box2(image):
    # immagine in gray scale
    print(image)
    # disegnare rettangolo
    #image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)
    coun, _= cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    pos= []
    for c in coun:
        # bbox = cv2.arcLength(c, True)
        x,y,w,h= cv2.boundingRect(c) # bbox
        if w>20 and h>20:
            dresspart= cv2.rectangle(image, (x,y), (x+w, y+h), 127, 2)
            pos.append([x, y, w, h])

    cv2.imshow('parte del vestito', dresspart)
    cv2.waitKey(0)
    return pos

def segmentation(img): # da mettere un .
    # pulire da rumore
    #img= noise(img)
    # result with 1 channels
    segment_mask1 = skimage.segmentation.felzenszwalb(img, scale=100) # 575 max value
    segment_mask2 = skimage.segmentation.felzenszwalb(img, scale=1000) # 100 max value

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(segment_mask1)
    ax1.set_xlabel("k=100")
    ax2.imshow(segment_mask2)
    ax2.set_xlabel("k=1000")
    fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
    plt.tight_layout()
    plt.show()

    plt.subplot(121), plt.title('Hist 1')
    plt.hist(segment_mask1)
    plt.subplot(122), plt.title('Hist 2')
    plt.hist(segment_mask2)
    plt.show()
    '''
    # cut mask1 consider only low values
    segment_mask1[segment_mask1>120]= 0
    plt.imshow(segment_mask1)
    plt.show()
    plt.hist(segment_mask1)
    plt.title('maschera1 tagliata nella parte superiore')
    plt.show()
    '''
    # TODO: come scegliere la scala considerando histogram
    c=segment_mask1 # con mask2
    c[c>80]=0 # c>30
    c[c < 40] = 0 # c<10
    plt.imshow(c) # per togliere bordi che non so: dilation, CLOSING (morphology)
    plt.show()
    c[c != 0] = 255  # {1}
    print('valori unici:\t',np.unique(c))
    c= np.array(c, dtype=np.float32)
    cv2.imshow('prova', c) # valuto il risultato ottenuto
    cv2.waitKey(0)

    # MORPHOLOGY
    clos= morp_closing(c)
    dil= morp_dilate(clos)

    pos= box2(dil) # oppure c
    return pos


if __name__ == "__main__":
    print('Inizio progetto')
    path = 'data-prova\\38814303li_0_e.jpg'
    path= 'data-prova\\12141279ui_0_r.jpg'
    path_dress='data-prova/12141279ui_1_f.jpg'
    # ---
    image, H, W, image_gray= openimage(path=path)  # immagine aperta nell'oggetto image
    image2 = cv2.imread('data-prova\\12141279ui_1_f.jpg')
    mask= cv2.imread('data-prova/12141279ui_0_r.png')

    # utili per la demo
    #extract_mask(mask,image)
    #dimension(mask)

    # detect della forma dell'abito nella situazione che ci sia solo lui
    #pos_image= dress(image2)

    # print(image.shape)
    # HIST
    #hist(image, 10)
    # SOGLIE
    #soglia(image) # thresh, lavora con gray images
    #soglia(image_gray)
    #sogliaRGB(image)
    # SOBEL
    #sobel(image)
    # CANNY
    #canny(image)
    # edge with colors
    #edgeHSV(image)
    # CONTOURS of the image
    # contour(image)
    # TEMPLATE MATCHING con le diverse dimensioni del template
    col = colors()
    template(image, image2, col)  # resize

    kmeans(image) # k-means sui colori

    bbox_position = segmentation(image)
    for x,y,w,h in bbox_position:
        dresspart = cv2.rectangle(image, (x, y), (x + w, y + h), 127, 2)
    cv2.imshow('parte del vestito', dresspart)
    cv2.waitKey(0)

    # Create display windows
    cv2.namedWindow("Edges")
    cv2.namedWindow("Image")
    # Trackbar will be used for changing threshold for edge
    initThresh = 105
    maxThresh = 200
    # Create trackbar to control the treshold
    cv2.createTrackbar("Threshold", "Image", initThresh, maxThresh, onTrackbarChange)
    onTrackbarChange(initThresh)

    # HOG with people detection
    hog(image)


''' parametri Canny
    image_gray= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    m = np.median(image_gray)
    sigma= 0.33
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
'''

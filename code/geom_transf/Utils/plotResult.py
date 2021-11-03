import sys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
from random import randint
import json

# define figure and axes
import torch
from PIL import Image
from PIL import ImageDraw

# data
column = ['Cloth', 'ClothMask', 'Image', 'ImageParse', 'Pose', 'WarpCloth', 'WarpMask', 'Result']
data_list = "test_pairs.txt"
data_root = "C:\\Users\\Eros Bignardi\\projectCV\\data"

# data list
cloth_names = []
cloth_mask_names = cloth_names
image_names = []
image_parse_names = []
image_pose_names = []
warp_cloth_names = cloth_names
warp_mask_names = cloth_names
result_names = image_names

with open(osp.join(data_root, data_list), 'r') as f:
    for line in f.readlines():
        im_name, c_name = line.strip().split()
        image_names.append(im_name)
        cloth_names.append(c_name)

        parse_name = im_name.replace('.jpg', '.png')
        image_parse_names.append(parse_name)

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        image_pose_names.append(pose_name)

print('len: ', len(cloth_names))

f, axarr = plt.subplots(5,8)
axarr[0, 0].set_title('Cloth')
axarr[0, 1].set_title('ClothMask')
axarr[0, 2].set_title('Image')
axarr[0, 3].set_title('ImageParse')
axarr[0, 4].set_title('Pose')
axarr[0, 5].set_title('WarpCloth')
axarr[0, 6].set_title('WarpMask')
axarr[0, 7].set_title('Result')

for i in range(5):
    value = randint(0,  len(cloth_names))
    # print('row: ', value + 1)

    # print("cloth")
    cloth = Image.open(osp.join(data_root, 'test\\cloth', cloth_names[value]))
    # print(cloth.filename)
    axarr[i, 0].imshow(cloth)

    # print("cloth mask")
    cloth_mask = Image.open(osp.join(data_root, 'test\\cloth-mask', cloth_mask_names[value]))
    # print(cloth_mask)
    axarr[i, 1].imshow(cloth_mask)

    # print("image")
    image = Image.open(osp.join(data_root, 'test\\image', image_names[value]))
    # print(image)
    axarr[i, 2].imshow(image)

    # print("parse")
    image_parse = Image.open(osp.join(data_root, 'test\\image-parse', image_parse_names[value]))
    # print(image_parse)
    axarr[i, 3].imshow(image_parse)

    # print("pose")
    # print(image_pose_names[i])
    # plot keypoints on image
    # axarr[i, 4].imshow(pose)

    # create a blank image where plot keypoints
    out = Image.new("RGB", (192, 256), (255, 255, 255))

    # load pose points
    with open(osp.join(data_root, 'test\\pose', image_pose_names[value]), 'r') as f:
        pose_label = json.load(f)
        # pose_data = pose_label['people'][0]['pose_keypoints']
        pose_data = pose_label['people'][-1]['pose_keypoints_2d']
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))
        # print(pose_data)

    point_num = pose_data.shape[0]
    pose_map = torch.zeros(point_num, 256, 192)
    r = 3
    im_pose = Image.new('L', (256, 192))
    pose_draw = ImageDraw.Draw(im_pose)
    for j in range(point_num):
        # out = Image.new('L', (256, 192))
        draw = ImageDraw.Draw(out)
        pointx = pose_data[j, 0]
        pointy = pose_data[j, 1]
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'black', 'black')

    axarr[i, 4].imshow(out)

    # print("warp cloth")
    warp_cloth = Image.open(osp.join(data_root, 'test\\warp-cloth', warp_cloth_names[value]))
    # print(warp_cloth)
    axarr[i, 5].imshow(warp_cloth)

    # print("warp mask")
    warp_mask = Image.open(osp.join(data_root, 'test\\warp-mask', warp_mask_names[value]))
    # print(warp_mask)
    axarr[i, 6].imshow(warp_mask)

    # print("result")
    result = Image.open(osp.join(data_root, 'test\\try-on', result_names[value]))
    # print(result)
    axarr[i, 7].imshow(result)
plt.show()



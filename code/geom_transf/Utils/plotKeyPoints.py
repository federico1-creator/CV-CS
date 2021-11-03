import numpy as np
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
image_names = []
image_pose_names = []

with open(osp.join(data_root, data_list), 'r') as f:
    for line in f.readlines():
        im_name, c_name = line.strip().split()
        image_names.append(im_name)

        pose_name = im_name.replace('.jpg', '_keypoints.json')
        image_pose_names.append(pose_name)

print('len: ', len(image_names))

f, axarr = plt.subplots(5, 2)
axarr[0, 0].set_title('Image')
axarr[0, 1].set_title('Pose')

for i in range(5):
    value = randint(0,  len(image_names))
    # print('row: ', value + 1)

    # print("image")
    image = Image.open(osp.join(data_root, 'test\\image', image_names[value]))
    # print(image)
    axarr[i, 0].imshow(image)

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

    axarr[i, 1].imshow(out)
plt.show()

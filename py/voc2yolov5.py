# -*- coding: utf-8 -*-

"""
@date: 2023/3/27 下午10:09
@file: voc2yolov5.py
@author: zj
@description: 
"""
from typing import List

import PIL
import os.path

import numpy as np
from PIL import Image

from tqdm import tqdm

import torchvision.datasets as datasets


def process(dataset: datasets.VOCDetection, cls_list: List, dst_root):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    dst_image_root = os.path.join(dst_root, 'images')
    if not os.path.exists(dst_image_root):
        os.makedirs(dst_image_root)
    dst_label_root = os.path.join(dst_root, 'labels')
    if not os.path.exists(dst_label_root):
        os.makedirs(dst_label_root)

    for idx in tqdm(range(len(dataset.images))):
        image, target = dataset.__getitem__(idx)
        img_w = int(target['annotation']['size']['width'])
        img_h = int(target['annotation']['size']['height'])

        label_list = list()
        for obj in target['annotation']['object']:
            difficult = int(obj['difficult'])
            if difficult != 0:
                continue
            cls_name = obj['name']
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])

            xcenter = (xmin + xmax) / 2
            ycenter = (ymin + ymax) / 2
            box_w = xmax - xmin
            box_h = ymax - ymin
            label_list.append(
                [cls_list.index(cls_name), xcenter / img_w, ycenter / img_h, box_w / img_w, box_h / img_h])

        # Save
        image_name = os.path.basename(dataset.images[idx])
        dst_img_path = os.path.join(dst_root, 'images', image_name)
        assert not os.path.exists(dst_img_path), dst_img_path
        assert isinstance(image, Image.Image)
        image.save(dst_img_path)

        label_name = os.path.splitext(image_name)[0] + '.txt'
        dst_label_path = os.path.join(dst_root, 'labels', label_name)
        assert not os.path.exists(dst_label_path), dst_label_path
        np.savetxt(dst_label_path, label_list, fmt='%f', delimiter=' ')


def create_yolov1_voc_dataset(data_root, cls_list, dst_root):
    print("=> Process train")
    dst_train_root = os.path.join(dst_root, 'yolov1-voc-train')
    dataset = datasets.VOCDetection(data_root, year='2012', image_set='train', download=True)
    process(dataset, list(cls_list), dst_train_root)

    dataset = datasets.VOCDetection(data_root, year='2007', image_set='train', download=True)
    process(dataset, list(cls_list), dst_train_root)

    dataset = datasets.VOCDetection(data_root, year='2007', image_set='test', download=True)
    process(dataset, list(cls_list), dst_train_root)

    print("=> Process val")
    dst_val_root = os.path.join(dst_root, 'yolov1-voc-val')
    dataset = datasets.VOCDetection(data_root, year='2012', image_set='val', download=True)
    process(dataset, list(cls_list), dst_val_root)


if __name__ == '__main__':
    data_root = '~/data/voc'
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    cls_list = np.loadtxt('../voc.names', dtype=str, delimiter=' ')
    print(cls_list)

    dst_root = './data'
    create_yolov1_voc_dataset(data_root, cls_list, dst_root)

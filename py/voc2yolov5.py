# -*- coding: utf-8 -*-

"""
@date: 2023/3/27 下午10:09
@file: voc2yolov5.py
@author: zj
@description:

Usage - Convert VOC dataset to YOLOv5:
    $ python py/voc2yolov5.py -s ../datasets/voc -d ../datasets/voc2yolov5-train -l trainval-2007 trainval-2012
    $ python py/voc2yolov5.py -s ../datasets/voc -d ../datasets/voc2yolov5-val -l test-2007

"""
import argparse
from typing import List

import os.path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.datasets as datasets

DELIMITER = '-'
SUPPORTS = ['train-2007', 'val-2007', 'test-2007', 'trainval-2007',
            'train-2012', 'val-2012', 'trainval-2012']


def parse_args():
    parser = argparse.ArgumentParser(description="VOC2YOLOv5")
    parser.add_argument('-s', '--src', metavar='SRC', type=str, help='Target Dataset Original Path.')
    parser.add_argument('-d', '--dst', metavar='DST', type=str, help='Target Dataset Result Path.')
    parser.add_argument("-l", '--list', nargs='+',
                        help='Specify dataset type and year. For example, test-2007、train-2012', required=True)

    parser.add_argument('--classes', metavar='CLASSES', type=str, default="voc.names",
                        help='Path of VOC classes')

    args = parser.parse_args()
    print("args:", args)
    return args


def process(dataset: datasets.VOCDetection, cls_list: List, dst_root: str):
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
            assert cls_name in cls_list, cls_name
            xmin = float(obj['bndbox']['xmin'])
            ymin = float(obj['bndbox']['ymin'])
            xmax = float(obj['bndbox']['xmax'])
            ymax = float(obj['bndbox']['ymax'])

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            box_w = xmax - xmin
            box_h = ymax - ymin
            # [x1, y1, x2, y2] -> [cls_id, x_center/img_w, y_center/img_h, box_w/img_w, box_h/img_h]
            label_list.append(
                [cls_list.index(cls_name), x_center / img_w, y_center / img_h, box_w / img_w, box_h / img_h])

        # Save
        image_name = os.path.basename(dataset.images[idx])
        dst_img_path = os.path.join(dst_image_root, image_name)
        assert not os.path.exists(dst_img_path), dst_img_path
        assert isinstance(image, Image.Image)
        image.save(dst_img_path)

        label_name = os.path.splitext(image_name)[0] + '.txt'
        dst_label_path = os.path.join(dst_label_root, label_name)
        assert not os.path.exists(dst_label_path), dst_label_path
        np.savetxt(dst_label_path, label_list, fmt='%f', delimiter=' ')


def main(args):
    data_root = os.path.abspath(args.src)
    dst_data_root = os.path.abspath(args.dst)

    cls_list = np.loadtxt(args.classes, dtype=str, delimiter=' ')
    print('cls_list:', cls_list)

    for item in args.list:
        assert item in SUPPORTS, item
        dataset_type, year = item.split(DELIMITER)
        print(f"Process Pascal VOC{year} {dataset_type}")

        dataset = datasets.VOCDetection(data_root, year=year, image_set=dataset_type, download=True)
        process(dataset, list(cls_list), dst_data_root)


if __name__ == '__main__':
    args = parse_args()
    main(args)

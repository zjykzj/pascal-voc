# -*- coding: utf-8 -*-

"""
@date: 2023/3/27 下午10:09
@file: voc2coco.py
@author: zj
@description:
python voc2coco.py -v /home/zj/data/voc -c /home/zj/data/voc/voc2coco -l train-2007 val-2007 test-2007 train-2012 val-2012
"""
import json
import os

import argparse
from typing import List

import sys
import os.path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torchvision.datasets as datasets

DELIMITER = '-'
SUPPORTS = ['train-2007', 'val-2007', 'test-2007', 'trainval-2007',
            'train-2012', 'val-2012', 'trainval-2007']


def parse_args():
    parser = argparse.ArgumentParser(description="VOC2COCO")
    parser.add_argument('-v', '--voc', metavar='VOC', type=str, help='Root Path of Pascal VOC Dataset.')
    parser.add_argument('-c', '--coco', metavar='COCO', type=str, help='Root Path of COCO-styled Dataset.')
    parser.add_argument("-l", '--list', nargs='+',
                        help='Specify dataset type and year. For example, test-2007、train-2012', required=True)

    args = parser.parse_args()
    print("args:", args)
    return args


def process(dataset: datasets.VOCDetection, cls_list: List, dst_root: str):
    if not os.path.exists(dst_root):
        os.makedirs(dst_root)
    dst_image_root = os.path.join(dst_root, 'images', f"{dataset.image_set}{dataset.year}")
    if not os.path.exists(dst_image_root):
        os.makedirs(dst_image_root)
    dst_annotations_root = os.path.join(dst_root, 'annotations')
    if not os.path.exists(dst_annotations_root):
        os.makedirs(dst_annotations_root)

    coco_anno_list = list()
    coco_image_list = list()

    bbox_id = 0
    for idx in tqdm(range(len(dataset.images))):
        image, target = dataset.__getitem__(idx)
        img_w = int(target['annotation']['size']['width'])
        img_h = int(target['annotation']['size']['height'])
        file_name = os.path.basename(dataset.images[idx])
        image_name = os.path.splitext(file_name)[0]

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

            box_w = xmax - xmin
            box_h = ymax - ymin

            anno_dict = dict()
            anno_dict['area'] = float(img_w * img_h)
            anno_dict['iscrowd'] = int(0)
            anno_dict['image_id'] = image_name
            anno_dict['bbox'] = [xmin, ymin, box_w, box_h]
            # 分类下标，从1开始
            anno_dict['category_id'] = cls_list.index(cls_name) + 1
            # 边界框id，每个边界框一个独立id
            anno_dict['id'] = bbox_id
            bbox_id += 1
            coco_anno_list.append(anno_dict)

        image_dict = dict()
        image_dict['file_name'] = file_name
        image_dict['height'] = img_h
        image_dict['width'] = img_w
        # 图片名。在coco数据集中，需要加上前缀`000000`，生成000000{id}.jpg
        image_dict['id'] = image_name
        coco_image_list.append(image_dict)

        # Save
        dst_img_path = os.path.join(dst_image_root, file_name)
        assert not os.path.exists(dst_img_path), dst_img_path
        assert isinstance(image, Image.Image)
        image.save(dst_img_path)

    coco_category_list = list()
    for idx, cls_name in enumerate(cls_list):
        category_dict = dict()
        category_dict['supercategory'] = cls_name
        # 等同于category_id
        category_dict['id'] = idx + 1
        category_dict['name'] = cls_name
        coco_category_list.append(category_dict)

    coco_anno_dict = dict()
    coco_anno_dict['images'] = coco_image_list
    coco_anno_dict['annotations'] = coco_anno_list
    coco_anno_dict['categories'] = coco_category_list

    annotation_path = os.path.join(dst_annotations_root, f'instances_{dataset.image_set}{dataset.year}.json')
    with open(annotation_path, 'w') as f:
        json.dump(coco_anno_dict, f)
    print(f"Save to {annotation_path}")


def main(args):
    data_root = os.path.abspath(args.voc)
    dst_data_root = os.path.abspath(args.coco)

    cls_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), '../voc.names')
    cls_list = np.loadtxt(cls_path, dtype=str, delimiter=' ')
    print('cls_list:', cls_list)

    for item in args.list:
        assert item in SUPPORTS, item
        dataset_type, year = item.split(DELIMITER)
        print(f"Process Pascal VOC {dataset_type} {year}")

        dataset = datasets.VOCDetection(data_root, year=year, image_set=dataset_type, download=True)
        process(dataset, list(cls_list), dst_data_root)


if __name__ == '__main__':
    args = parse_args()
    print('args:', args)
    main(args)

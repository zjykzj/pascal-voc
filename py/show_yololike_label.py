# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:22
@File    : show_yololike_image.py
@Author  : zj
@Description:

The parameter `image` can be a file or a directory, and the parameter `label` must correspond to.

When specified as a directory, the program will first search for all images, and then search for corresponding file in the `label`

Usage: Show image with VOCLike label:
    $ python3 py/show_yololike_label.py assets/yololike/000000082986.jpg assets/yololike/000000082986.txt
    $ python3 py/show_yololike_label.py assets/yololike/ assets/yololike/

Usage: Save annotated image:
    $ python3 py/show_yololike_label.py assets/yololike/000000082986.jpg assets/yololike/000000082986.txt --dst ./output/
    $ python3 py/show_yololike_label.py assets/yololike/ assets/yololike/ --dst ./output/

"""

from typing import Dict, List, Any, Tuple

import os
import glob
import argparse
from argparse import Namespace

import cv2
import numpy as np
from numpy import ndarray


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Show VOCLike label")
    parser.add_argument('image', metavar='IMAGE', type=str,
                        help='Image path, can be a file path or a directory.')
    parser.add_argument("label", metavar='LABEL', type=str,
                        help="YOLOLike label path, can be a file path or a directory.")

    parser.add_argument('--dst', metavar='DST', type=str, default=None,
                        help='Save data dir.')
    args = parser.parse_args()
    print("args:", args)
    return args


def parse_yolo_txt(label_path: str) -> List:
    target = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            items = line.strip().split(' ')
            if len(items) != 5:
                continue

            target.append(np.array(items, dtype=float))
    return target


def show_image_label(image_path: str, label_path: str) -> Tuple[ndarray, str]:
    # Image
    assert os.path.isfile(image_path), image_path
    image = cv2.imread(image_path)
    # Label
    assert os.path.isfile(label_path), label_path
    target = parse_yolo_txt(label_path)
    print(target)

    image_h, image_w = image.shape[:2]
    for items in target:
        cls_id, xc, yc, box_w, box_h = items
        print(cls_id, xc, yc, box_w, box_h)

        xmin = int((xc - box_w / 2) * image_w)
        ymin = int((yc - box_h / 2) * image_h)
        xmax = int((xc + box_w / 2) * image_w)
        ymax = int((yc + box_h / 2) * image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    image_name = os.path.basename(image_path)
    return image, image_name


def main(args: Namespace):
    image_list = []
    label_list = []
    if os.path.isfile(args.image) and os.path.isfile(args.label):
        image_list.append(args.image)
        label_list.append(args.label)
    elif os.path.isdir(args.image) and os.path.isdir(args.label):
        image_dir = args.image
        label_dir = args.label
        for image_path in (glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))):
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{image_name}.txt")
            if not os.path.exists(label_path):
                continue

            image_list.append(image_path)
            label_list.append(label_path)
    else:
        raise ValueError("Please provide correct args.image and args.label")

    dst_dir = args.dst
    for image_path, label_path in zip(image_list, label_list):
        image, image_name = show_image_label(image_path, label_path)

        cv2.imshow("image", image)
        cv2.waitKey(0)

        if dst_dir is not None:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            assert os.path.isdir(dst_dir), f"{dst_dir} is not a directory"

            dst_image_path = os.path.join(dst_dir, image_name)

            cv2.imwrite(dst_image_path, image)
            print(f"Save to {dst_image_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

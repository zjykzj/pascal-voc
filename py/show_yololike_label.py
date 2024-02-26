# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:22
@File    : show_yololike_image.py
@Author  : zj
@Description:

Usage: Show image with YOLOLike label:
    $ python3 py/show_yololike_label.py ./assets/yololike/000000082986.jpg ./assets/yololike/000000082986.txt

Usage: Save annotated image:
    $ python3 py/show_yololike_label.py ./assets/yololike/000000082986.jpg ./assets/yololike/000000082986.txt --dst ./output/

Usage: Show Multi-images:
    $ python3 py/show_yololike_label.py ./assets/yololike/ ./assets/yololike/

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
    dst_dir = args.dst
    if os.path.isfile(args.image) and os.path.isfile(args.label):
        image_list = [args.image]
        label_list = [args.label]
    else:
        assert os.path.isdir(args.image) and os.path.isdir(args.label)
        image_list = []
        label_list = []
        for image_path in (glob.glob(os.path.join(args.image, "*.jpg")) + glob.glob(os.path.join(args.image, "*.png"))):
            image_name = os.path.basename(image_path)
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(args.label, label_name)
            assert os.path.isfile(image_path) and os.path.isfile(label_path), print(
                f"Image path: {image_path}, Label path: {label_path}")
            image_list.append(image_path)
            label_list.append(label_path)

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

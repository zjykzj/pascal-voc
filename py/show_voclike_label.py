# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:22
@File    : show_voclike_image.py
@Author  : zj
@Description:

The parameter `image` can be a file or a directory, and the parameter `label` must correspond to.

When specified as a directory, the program will first search for all images, and then search for corresponding file in the `label`

Usage: Show image with VOCLike label:
    $ python3 py/show_voclike_label.py assets/voclike/000006.jpg assets/voclike/000006.xml
    $ python3 py/show_voclike_label.py assets/voclike/ assets/voclike/

Usage: Save annotated image:
    $ python3 py/show_voclike_label.py assets/voclike/000006.jpg assets/voclike/000006.xml --dst ./output/
    $ python3 py/show_voclike_label.py assets/voclike/ assets/voclike/ --dst ./output/

"""
import glob
from typing import Dict, List, Any

import os
import argparse
import collections

import cv2
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description="Show VOCLike label")
    parser.add_argument('image', metavar='IMAGE', type=str, default="",
                        help='Image path.')
    parser.add_argument("label", metavar='LABEL', type=str, default="",
                        help="VOCLike label path or dir.")

    parser.add_argument('--dst', metavar='DST', type=str, default=None,
                        help='Save data dir.')
    args = parser.parse_args()
    print("args:", args)
    return args


def parse_voc_xml(node: ET.Element) -> Dict[str, Any]:
    voc_dict: Dict[str, Any] = {}
    children = list(node)
    if children:
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    if node.text:
        text = node.text.strip()
        if not children:
            voc_dict[node.tag] = text
    return voc_dict


def main(args):
    image_list = []
    label_list = []
    if os.path.isfile(args.image) and os.path.isfile(args.label):
        image_list.append(args.image)
        label_list.append(args.label)
    elif os.path.isdir(args.image) and os.path.isdir(args.label):
        image_dir = args.image
        label_dir = args.label
        for image_path in glob.glob(os.path.join(image_dir, "*.jpg")):
            if not image_path.endswith('.jpg'):
                continue
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{image_name}.xml")
            if not os.path.exists(label_path):
                continue

            image_list.append(image_path)
            label_list.append(label_path)
    else:
        raise ValueError("Please provide correct args.image and args.label")

    dst_dir = args.dst
    for image_path, label_path in zip(image_list, label_list):
        # Image
        assert os.path.isfile(image_path), image_path
        image = cv2.imread(image_path)
        # Label
        assert os.path.isfile(label_path), label_path
        target = parse_voc_xml(ET.parse(label_path).getroot())
        print(target)

        for object in target['annotation']['object']:
            # print(object['bndbox'])
            xmin = int(object['bndbox']['xmin'])
            ymin = int(object['bndbox']['ymin'])
            xmax = int(object['bndbox']['xmax'])
            ymax = int(object['bndbox']['ymax'])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

            category = object['name']
            cv2.putText(image, category, (xmin, ymin - 10), 0, 0.5, (0, 255, 0), 1)

        cv2.imshow("image", image)
        cv2.waitKey(0)

        if dst_dir is not None:
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            assert os.path.isdir(dst_dir), f"{dst_dir} is not a directory"

            image_name = os.path.basename(image_path)
            dst_image_path = os.path.join(dst_dir, image_name)

            cv2.imwrite(dst_image_path, image)
            print(f"Save to {dst_image_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

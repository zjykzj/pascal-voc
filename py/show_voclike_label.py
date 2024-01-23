# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 20:22
@File    : show_voclike_image.py
@Author  : zj
@Description:

Usage: Show image with VOCLike label:
    $ python3 py/show_voclike_label.py assets/voclike/000006.jpg assets/voclike/000006.xml

Usage: Save annotated image:
    $ python3 py/show_voclike_label.py assets/voclike/000006.jpg assets/voclike/000006.xml --dst ./output/

"""

from typing import Dict, List, Any

import os
import argparse
import collections

import cv2
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description="Show VOCLike label")
    parser.add_argument('image', metavar='IMAGE', type=str,
                        help='Image path.')
    parser.add_argument("label", metavar='LABEL', type=str,
                        help="VOCLike label path.")

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
    image_path = args.image
    label_path = args.label
    dst_dir = args.dst

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

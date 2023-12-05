# -*- coding: utf-8 -*-

"""
@Time    : 2023/11/13 20:08
@File    : voclike2yolov5.py
@Author  : zj
@Description:

Usage: Parse VOC XML:
    $ python3 py/voclike2yolov5.py -x assets/voclike/000006.xml -c assets/voclike/classes

"""

import argparse
import collections

from typing import Dict, List, Any
import xml.etree.ElementTree as ET

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="VOCLike2YOLOv5")
    parser.add_argument('-x', '--xml', metavar='XML', type=str, default='assets/voclike/000006.xml',
                        help='VOCLike XML path.')
    parser.add_argument('-c', '--classes', metavar='CLASSES', type=str, default='assets/voclike/classes',
                        help='VOCLike CLASS path.')

    args = parser.parse_args()
    print("args:", args)
    return args


def voc2yolov5_label(target: Dict, cls_list: List):
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

    return label_list


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


if __name__ == '__main__':
    args = parse_args()
    target = parse_voc_xml(ET.parse(args.xml).getroot())
    print(f"target: {target}")

    classes = np.loadtxt(args.classes, dtype=str, delimiter=" ").tolist()
    if isinstance(classes, str):
        classes = [classes]
    print(f"classes: {classes}")
    label_list = voc2yolov5_label(target, classes)
    print(f"label_list: {label_list}")

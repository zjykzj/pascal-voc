# -*- coding: utf-8 -*-

"""
@Time    : 2024/1/23 19:55
@File    : find_classes.py
@Author  : zj
@Description:

Usage: Traverse all label files, obtain category list and save:
    $ python3 py/find_classes.py ../../myai/mask/datasets/MaskDatasets/datasets/

"""

from typing import Dict, List, Any

import os
import argparse
import collections

import numpy as np
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description="Find Classes")
    parser.add_argument('label', metavar='LABEL', type=str,
                        help='VOCLike data root path.')

    parser.add_argument('--dst', metavar='DST', type=str, default='./output',
                        help='Save data dir.')
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


def load_voc_data(root):
    assert os.path.isdir(root), root

    xml_list = list()
    print(f"Retrieval {root}")
    for xml_path in Path(root).rglob(pattern="*.xml"):
        xml_list.append(xml_path)

    return xml_list


def main(args):
    class_list = list()

    label_dir = args.label
    xml_list = load_voc_data(label_dir)
    for xml_path in tqdm(xml_list):
        # Label
        target = parse_voc_xml(ET.parse(xml_path).getroot())
        for object in target['annotation']['object']:
            if object['name'] not in class_list:
                class_list.append(object['name'])

    class_list = sorted(class_list)
    print(f"Found classes: {class_list}")

    save_root = args.dst
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    assert os.path.isdir(save_root), save_root
    class_path = os.path.join(save_root, "classes.txt")
    np.savetxt(class_path, class_list, delimiter=" ", fmt='%s')
    print(f"Save to {class_path}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

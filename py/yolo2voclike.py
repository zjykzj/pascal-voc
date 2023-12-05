# -*- coding: utf-8 -*-

"""
@Time    : 2023/12/5 9:58
@File    : yolo2voclike.py
@Author  : zj
@Description: 
"""

import os
import glob
import argparse
import shutil
import xmltodict
import cv2

from tqdm import tqdm
import numpy as np

XML_SAMPLE = "assets/voclike/000006.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="VOCLike2YOLOv5")
    parser.add_argument('src', metavar='SRC', type=str,
                        help='YOLOv5 data root path.')
    parser.add_argument("classes", metavar='CLASSES', type=str,
                        help="Classes path.")

    parser.add_argument('dst', metavar='DST', type=str,
                        help='VOCLike data root path.')

    args = parser.parse_args()
    print("args:", args)
    return args


def load_yolo_data(root):
    assert os.path.isdir(root), root

    image_root = os.path.join(root, "images")
    assert os.path.isdir(image_root), image_root
    label_root = os.path.join(root, "labels")
    assert os.path.isdir(label_root), label_root

    image_list = list()
    label_list = list()
    for label_path in glob.glob(os.path.join(label_root, "*.txt")):
        image_path = label_path.replace("labels", "images").replace(".txt", ".jpg")
        if os.path.isfile(image_path):
            image_list.append(image_path)
            label_list.append(label_path)

    return image_list, label_list


def parse_yolo_to_voc(image_path, label_path, classes):
    with open(XML_SAMPLE, 'rb') as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    image_name = os.path.basename(image_path)
    data_dict['annotation']['filename'] = image_name
    data_dict['annotation']['path'] = image_path

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_height, img_width = image.shape[:2]
    data_dict['annotation']['size']['width'] = img_width
    data_dict['annotation']['size']['height'] = img_height

    object_list = list()
    label_list = np.loadtxt(label_path, dtype=float, delimiter=' ')
    for label in label_list:
        cls_id, x_c, y_c, box_w, box_h = label
        class_name = classes[int(cls_id)]
        x_min = int((x_c - box_w / 2) * img_width)
        y_min = int((y_c - box_h / 2) * img_height)
        x_max = int((x_c + box_w / 2) * img_width)
        y_max = int((y_c + box_h / 2) * img_height)
        object_list.append({
            'name': class_name,
            'pose': 'Unspecified',
            'truncated': '0',
            'difficult': '0',
            'bndbox': {'xmin': str(x_min), 'ymin': str(y_min), 'xmax': str(x_max), 'ymax': str(y_max)}
        })
    data_dict['annotation']['object'] = object_list

    return data_dict


def main(args):
    save_root = args.dst
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    class_path = args.classes
    dst_class_path = os.path.join(save_root, os.path.basename(class_path))
    shutil.copyfile(class_path, dst_class_path)
    classes = np.loadtxt(class_path, dtype=str, delimiter=' ').tolist()
    if isinstance(classes, str):
        classes = [classes]

    image_list, label_list = load_yolo_data(args.src)
    for image_path, label_path in tqdm(zip(image_list, label_list), total=len(image_list)):
        # Image
        image_name = os.path.basename(image_path)
        dst_image_path = os.path.join(save_root, image_name)
        shutil.copyfile(image_path, dst_image_path)

        # Label
        data_dict = parse_yolo_to_voc(image_path, label_path, classes)
        xml_string = xmltodict.unparse(data_dict, pretty=True)

        label_name = os.path.basename(label_path).replace(".txt", ".xml")
        dst_label_path = os.path.join(save_root, label_name)
        with open(dst_label_path, 'w') as f:
            f.write(xml_string)

    print(f"Save to {save_root}")


if __name__ == '__main__':
    args = parse_args()
    main(args)

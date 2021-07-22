"""
split test annotations according to class name
and save at data/class_annotation/[classname].txt
each line: 'img_idx x1,y1,x2,y2,difficult'
"""
import sys
import os
import os.path as osp

Base_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(Base_dir, ".."))
from collections import defaultdict
from config.yolov4_config import DATASET_PATH
import xml.etree.ElementTree as ET
ANNO_PATH = osp.join(DATASET_PATH, 'Annotations')


def parse_anno(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)
    return objects



if __name__ == "__main__":
    cls_record_path = osp.join(DATASET_PATH, "ClassAnnos")
    if not os.path.exists(cls_record_path):
        os.mkdir(cls_record_path)

    with open(osp.join(DATASET_PATH, 'test.txt')) as f:
        img_list = f.readlines()

    Cls_Record = defaultdict(list)
    for img_idx in img_list:
        img_idx = img_idx.strip()
        objs = parse_anno(osp.join(ANNO_PATH, img_idx+'.xml'))
        line = img_idx + ' {:d},{:d},{:d},{:d},{:d}\n'
        for obj in objs:
            cls_name = obj["name"]
            difficult = obj["difficult"]
            bbox = obj["bbox"]
            Cls_Record[cls_name].append(line.format(*bbox, difficult))

    for cls_name, lines in Cls_Record.items():
        with open(osp.join(cls_record_path, cls_name+'.txt'), 'w') as f:
            f.write(''.join(lines))




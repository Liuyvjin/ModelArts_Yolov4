# coding=utf-8
import os.path as osp
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))
DATA_PATH = osp.join(PROJECT_PATH, 'data')
WEIGHT_PATH = osp.join(PROJECT_PATH, 'weights')

DATASET_NAME = "speedstar_dataset"  # include: JPEGImages/  Annotations/  test.txt   train.txt
DATASET_PATH = osp.join(DATA_PATH, DATASET_NAME)

MODEL_TYPE = {
    "TYPE": "Mobilenetv3-YOLOv4"
}  # YOLO type:YOLOv4, Mobilenet-YOLOv4, CoordAttention-YOLOv4 or Mobilenetv3-YOLOv4

CONV_TYPE = {"TYPE": "DO_CONV"}  # conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE

# train
TRAIN = {
    "DATA_TYPE": "Customer",  # DATA_TYPE: VOC ,COCO or Customer
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 12,
    "MULTI_SCALE_TRAIN": True,
    "IOU_THRESHOLD_LOSS": 0.5,
    "YOLO_EPOCHS": 50,
    "Mobilenet_YOLO_EPOCHS": 200,
    "EVAL_EPOCHS": 90,
    "NUMBER_WORKERS": 3,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 2,  # or None
    "showatt": False
}


# val
VAL = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 1,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.005,
    "NMS_THRESH": 0.45,
    "MULTI_SCALE_VAL": False,
    "FLIP_VAL": False,
    "Visual": False,
    "showatt": False
}

Customer_DATA = {
    "NUM": 6,  # your dataset number
    "CLASSES": [
        "red_stop",
        "green_go",
        "yellow_back",
        "speed_limited",
        "speed_unlimited",
        "pedestrian_crossing",
        ],  # your dataset class
}

# model
MODEL = {
    "ANCHORS": [
        [
            [ 3.25,     6.375  ],
            [ 4.375 ,   9.5    ],
            [ 6. ,      8.125  ],
        ],  # Anchors for small obj
        [
            [ 3.25,     6.5625 ],
            [ 4.8125,   8.3125 ],
            [ 6.625 ,  12.3125 ],
        ],  # Anchors for medium obj
        [
            [ 5.40625 , 8.4375 ],
            [ 6.4375,   0.5625 ],
            [13.0625 ,  1.65625],
        ],
    ],  # Anchors for big obj
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}


# [[ 3.25 ,    6.375  ],
#  [ 4.375,    9.5    ],
#  [ 6. ,      8.125  ],
#  [ 3.25,     6.5625 ],
#  [ 4.8125,   8.3125 ],
#  [ 6.625 ,  12.3125 ],
#  [ 5.40625 , 8.4375 ],
#  [ 6.4375,   0.5625 ],
#  [13.0625 ,  1.65625],]
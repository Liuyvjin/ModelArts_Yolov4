import os.path as osp
import torch
import numpy as np
from eval.evaluator import Evaluator
from model.build_model import Build_Model
from config.yolov4_config import WEIGHT_PATH, DATASET_PATH
from time import time
import cv2

if __name__ == '__main__':
    img_path = osp.join(DATASET_PATH, 'JPEGImages') + '\\{:s}.jpg'
    test_file = osp.join(DATASET_PATH, 'test.txt')
    # load model
    weight_path = osp.join(WEIGHT_PATH, 'best.pth')
    print("loading weight file from : {}".format(weight_path))
    yolo = Build_Model().cuda()
    yolo.load_state_dict(torch.load(weight_path))

    # predict
    predictor = Evaluator(yolo, conf_thresh=0.46)
    start = time()
    with open(test_file, 'r') as f:
        img_list = f.readlines()
    img_list = np.array([img_name.strip() for img_name in img_list])
    np.random.shuffle(img_list)

    cv2.namedWindow('Predict', flags=cv2.WINDOW_NORMAL)
    for img_name in img_list:
        # read image
        img = cv2.imread(img_path.format(img_name))
        print('Show: ' + img_path.format(img_name))
        print('Continue? ([y]/n)? ')
        pred = predictor.predict(img)
        img = predictor.visualize(img, pred)
        cv2.imshow('Predict', img)
        c = cv2.waitKey()
        if c in [ord('n'), ord('N')]:
            exit()


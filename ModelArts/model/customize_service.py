from model_service.pytorch_model_service import PTServingBaseService
import torch.nn as nn
import torch
import json
import numpy as np
import torchvision.transforms as transforms
import cv2
from build_model import Build_Model
from tools import *
from yolov4_config import Customer_DATA

import os.path as osp

class Yolov4Service(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(Yolov4Service, self).__init__(model_name, model_path)
        self.base_dir =  osp.dirname(osp.realpath(__file__))
        # load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = osp.join(self.base_dir, 'model_backup.pth')
        self.yolo = Build_Model().to(self.device)
        self.yolo.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.yolo.eval()

        self.test_size = 416
        self.org_shape = (0, 0)
        self.conf_thresh = 0.46
        self.nms_thresh = 0.45
        self.Classes = np.array(Customer_DATA["CLASSES"])

    def _preprocess(self, data):
        # 预处理成{key: input_batch_var}，input_batch_var为模型输入张量
        pro_data = {}
        image_dict = data['images']
        input_batch = []
        for img_name, img_content in image_dict.items():
            img = self.__imread(img_content)
            self.org_shape = img.shape[:2]
            img = self.__transform(img)
            input_batch.append(img)
        pro_data['images'] = torch.stack(input_batch, dim=0)

        return pro_data

    def _inference(self, data):
        result = {}
        with torch.no_grad():
            input_batch = data['images']
            # v - [1, C, test_size, test_size]
            _, pred = self.yolo(input_batch)
            pred = pred.squeeze().cpu().numpy()
            # pred - [N, 6(xmin, ymin, xmax, ymax, score, class)]
            pred = self.resize_filter_pb( pred, self.test_size, self.org_shape)
            result['images'] = nms(pred, self.conf_thresh, self.nms_thresh)
        return result

    def _postprocess(self, data):
        # 根据标签索引到图片的分类结果
        result = {}
        for k, v in data.items():
            cls_idx = v[:, 5].astype(np.int32)
            detection_classes = self.Classes[cls_idx]
            detection_boxes = v[:, [1,0,3,2]].round().astype(np.int32)
            detection_scores = v[:, 4]
            result = {
                'detection_classes': detection_classes.tolist(),
                'detection_boxes'  : detection_boxes.tolist(),
                'detection_scores' : detection_scores.tolist()
            }
        return result

    def resize_filter_pb( self, pred_bbox, test_input_size, org_img_shape, valid_scale=(0, np.inf) ):
        """
        input: pred_bbox - [Sigma0~2[batchsize x grid[i] x grid[i] x anchors(3)], x+y+w+h+conf+cls_6 (11)]
        Resize to origin size and Filter out the prediction box to remove the unreasonable scale of the box
        output: pred_bbox [n, x+y+w+h+conf+cls (6)]
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4]) # to xyxy
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0*test_input_size/org_w, 1.0*test_input_size/org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)Crop off the portion of the predicted Bbox that is beyond the original image
        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )
        # (3)Sets the coor of an invalid bbox to 0
        invalid_mask = np.logical_or(
            (pred_coor[:, 0] > pred_coor[:, 2]),
            (pred_coor[:, 1] > pred_coor[:, 3]),
        )
        pred_coor[invalid_mask] = 0

        # (4)Remove bboxes that are not in the valid range
        # for [0, inf], this can be skiped
        bboxes_scale = np.sqrt(
            np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1)
        )
        scale_mask = np.logical_and(
            (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])
        )

        # (5)Remove bboxes whose score is below the score_threshold
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate(
            [coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1
        )

        return bboxes

    def __transform(self, img):
        img = Resize((self.test_size, self.test_size), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img).float().to(self.device)
        # return torch.from_numpy(img[np.newaxis, ...]).float().to(self.device)

    def __imread(self, img_byteio):
        return cv2.imdecode(
            np.frombuffer(img_byteio.getbuffer(), np.uint8),
            cv2.IMREAD_COLOR
        )





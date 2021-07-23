import sys
import os
import os.path as osp
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm
from collections import defaultdict

Base_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(Base_dir, ".."))
from utils.data_augment import Resize
from eval.voc_eval import voc_eval
from utils.tools import nms, xywh2xyxy
from utils.visualize import visualize_boxes
import config.yolov4_config as cfg
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
current_milli_time = lambda: int(round(time.time() * 1000))


class Evaluator(object):
    def __init__(self, model=None, conf_thresh=None):
        self.classes        =   cfg.Customer_DATA["CLASSES"]    # default: use customer dataset
        if conf_thresh is None:
            self.conf_thresh =  cfg.VAL["CONF_THRESH"]          # 0.005
        else:
            self.conf_thresh = conf_thresh
        self.nms_thresh     =   cfg.VAL["NMS_THRESH"]           # 0.45
        self.test_size      =   cfg.VAL["TEST_IMG_SIZE"]        # 416

        self.model          =   model
        self.device         =   next(model.parameters()).device
        self.class_record   =   defaultdict(list)               # detected objs categorized by classes

        self.max_visual_img =   20 # max num of pred images to be visualized
        self.cnt_visual_img =   0  # num of pred images be visualized
        self.inference_time =   0.0

        self.pred_image_path    = osp.join(cfg.DATA_PATH, "pred_images")

    def calc_APs(self):
        if not osp.exists(self.pred_image_path):
            os.mkdir(self.pred_image_path)
        # read images from test.txt
        img_list_file = osp.join( cfg.DATASET_PATH,  "test.txt" )
        with open(img_list_file, "r") as f:
            lines = f.readlines()
            img_list = [line.strip() for line in lines]
        # predict all imgs, save result in class_record
        pool = ThreadPool(multiprocessing.cpu_count())
        with tqdm(total=len(img_list), ncols=120, smoothing=0.9 ) as tq:
            for i, _ in enumerate(pool.imap_unordered(self.predict_and_save, img_list)):
                tq.update()
        self.inference_time = 1.0 * self.inference_time / len(img_list)
        # calc mAP
        APs = {}
        for cls, data in self.class_record.items():
            img_idxs = np.array([rec[0] for rec in data])
            bbox_conf = np.array([rec[1] for rec in data], dtype=np.float)
            _, _, AP = voc_eval((img_idxs, bbox_conf), cls_name=cls)
            APs[cls] = AP
        return APs, self.inference_time

    def predict_and_save(self, img_idx):
        # find img file
        img_path = osp.join(cfg.DATASET_PATH, 'JPEGImages', img_idx + '.jpg')
        img = cv2.imread(img_path)
        # predict all valid bboxes in img [N, 6]
        bboxes_prd = self.predict(img)
        # visualization
        if bboxes_prd.shape[0] != 0  and self.cnt_visual_img < self.max_visual_img:
            self.visualize(img, bboxes_prd, img_idx, save=True)
            self.cnt_visual_img += 1

        # save to class_record
        for bbox in bboxes_prd:
            cls_name = self.classes[int(bbox[5])]
            self.class_record[cls_name].append([img_idx, bbox[:5]])

    def predict(self, img):
        org_shape = img.shape[:2]
        img = self.__transform(img)  # [1, C, test_size, test_size]
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            _, pred_decode = self.model(img)   # p_d
            self.inference_time += current_milli_time() - start_time
        # p_d: [Sigma0~2[batchsize x grid[i] x grid[i] x anchors(3)], x+y+w+h+conf+cls_6(11)]
        pred_decode = pred_decode.squeeze().cpu().numpy()
        # [N, 6(xmin, ymin, xmax, ymax, score, class)]
        bboxes = self.resize_filter_pb( pred_decode, self.test_size, org_shape)
        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)
        return bboxes

    def resize_filter_pb(
        self, pred_bbox, test_input_size, org_img_shape, valid_scale=(0, np.inf)
    ):
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
        return torch.from_numpy(img[np.newaxis, ...]).float().to(self.device)

    def visualize(self, img, pred, img_name=None, save=False):
        if len(pred) > 0:
            bboxes  = pred[:, :4]
            cls_ids = pred[:, 5].round().astype(np.int32)
            scores  = pred[:, 4]
            visualize_boxes(
                image=img, boxes=bboxes, labels=cls_ids, probs=scores,
                class_labels=self.classes, min_score_thresh=self.conf_thresh)
        if save:
            save_path = osp.join(self.pred_image_path, "{}.jpg".format(img_name))
            cv2.imwrite(save_path, img)
        return img


if __name__ == '__main__':
    import time
    from model.build_model import Build_Model

    weight_path = osp.join(Base_dir, '../weights/best.pth')
    print("loading weight file from : {}".format(weight_path))
    chkpt = torch.load(weight_path, map_location=torch.device('cuda'))
    yolo = Build_Model().cuda()
    yolo.load_state_dict(chkpt)
    del chkpt
    evalutaor = Evaluator(yolo)

    start = time.time()
    mAP = 0
    Aps, _ = evalutaor.calc_APs()
    for k, v in Aps.items():
        print('class {}: {:.3f}'.format(k, v))
        mAP += v
    print('mAP: {:.3f}'.format(mAP/6.0))
    print('total time: %d' % (time.time() - start))


import os.path as osp
import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.visualize import *
from utils.torch_utils import *
from utils.logger import Logger


class Evaluation(object):
    def __init__(
        self,
        logger: Logger,
        gpu_id=0,
        weight_path=None,
        img_path=None,
        mode=None
    ):
        self.logger         = logger
        self.num_class      = cfg.Customer_DATA["NUM"]
        self.conf_threshold = cfg.VAL["CONF_THRESH"]
        self.nms_threshold  = cfg.VAL["NMS_THRESH"]
        self.device         = gpu.select_device(gpu_id)
        self.showatt        = cfg.VAL["showatt"]
        self.img_path       = img_path
        self.mode           = mode
        self.classes        = cfg.Customer_DATA["CLASSES"]

        self.model = Build_Model(showatt=self.showatt).to(self.device)

        self.__load_model_weights(weight_path)

        self.evalutaor = Evaluator(self.model, showatt=self.showatt)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))
        chkpt = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(chkpt["model"])
        del chkpt

    def val(self):
        self.logger.info(" ==== Start Evaluation ==== ")
        start = time.time()
        mAP = 0
        with torch.no_grad():
            APs, inference_time = Evaluator( self.model, showatt=False ).APs_voc()  ### 每一个epoch都要生成一个就离谱
            for cls in APs:
                logger.info("mAP of class: {} = {}".format(cls, APs[cls]))
                mAP += APs[cls]
            mAP = mAP / self.num_class
            logger.info("mAP: {}".format(mAP))
            logger.info("inference time: {:.2f} ms".format(inference_time))
        end = time.time()
        logger.info("  Val cost time:{:.4f}s".format(end - start))

    def detection(self):
        if self.img_path:
            imgs = os.listdir(self.img_path)
            logger.info(" ==== Start Detection ==== ")
            for img_name in imgs:
                path = os.path.join(self.img_path, img_name)
                logger.info("val images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.evalutaor.get_bbox(img, img_name, mode=self.mode)
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(
                        image=img,
                        boxes=boxes,
                        labels=class_inds,
                        probs=scores,
                        class_labels=self.classes,
                    )
                    path = os.path.join(
                        cfg.DATA_PATH, "pred_imgaes/{}".format(img_name)
                    )

                    cv2.imwrite(path, img)
                    logger.info("saved images : {}".format(path))


if __name__ == "__main__":
    global logger
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_path",    type=str,   default="best.pth",   help="weight file path")
    parser.add_argument("--log_val_path",   type=str,   default="logs_val/",          help="val log file path")
    parser.add_argument("--gpu_id",         type=int,   default=-1, help="whither use GPU(eg:0,1,2) or CPU(-1)")
    parser.add_argument("--img_path",       type=str,   default=cfg.DATASET_PATH+'\\JPEGImages', help="det data path or None")
    parser.add_argument("--mode",           type=str,   default="val",              help="val or det")
    args = parser.parse_args()

    log_val_path = osp.join(cfg.PROJECT_PATH, args.log_val_path)
    weight_path  = osp.join(cfg.WEIGHT_PATH, args.weight_path)

    logger = Logger(
        log_file_name   =   osp.join(log_val_path, "/run_voc_val.log"),
        logger_name     =   "YOLOv4",
    )

    if args.mode == "val":
        Evaluation(
            logger      =   logger,
            gpu_id      =   args.gpu_id,
            weight_path =   args.weight_path,
            mode        =   args.mode
        ).val()
    else:
        Evaluation(
            logger      =   logger,
            gpu_id      =   args.gpu_id,
            weight_path =   args.weight_path,
            img_path    =   args.img_path,
            mode        =   args.mode
        ).detection()

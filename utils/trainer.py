import os.path as osp
import sys
FILE_DIR = osp.dirname(osp.abspath(__file__))
PROJ_DIR = osp.join(FILE_DIR, '..')
sys.path.append(PROJ_DIR)
import config.yolov4_config as cfg
from model.build_model import Build_Model
from model.loss.yolo_loss import YoloV4Loss
import utils.gpu as gpu
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import utils.datasets as data
import time
import random
from eval.evaluator import *
from utils.tools import *
from tensorboardX import SummaryWriter
from utils.cosine_lr_scheduler import CosineDecayLR
from utils.logger import Logger
from apex import amp
from utils.torch_utils import init_seeds
from tqdm import tqdm


def is_valid_number(x):
    return not (math.isnan(x) or math.isinf(x) or x > 1e4)


class Trainer(object):

    def __init__(   self,
                    logger: Logger,
                    writer: SummaryWriter,
                    weight_file =   None,
                    resume      =   False,
                    gpu_id      =   0,
                    accumulate  =   1,
                    fp_16       =   False  ):

        init_seeds(0)
        self.fp_16 = fp_16
        self.device = gpu.select_device(gpu_id)
        self.start_epoch = 0
        self.best_mAP = 0.0
        self.accumulate = accumulate
        self.weight_path = osp.join(cfg.WEIGHT_PATH, weight_file)
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        self.showatt = cfg.TRAIN["showatt"]
        self.logger  = logger
        self.writer  = writer
        if self.multi_scale_train:
            print("Using multi scales training")
        else:
            print("train img size is {}".format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.epochs = (
            cfg.TRAIN["YOLO_EPOCHS"]
            if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
            else cfg.TRAIN["Mobilenet_YOLO_EPOCHS"]
        )
        self.eval_epoch = ( 30  if cfg.MODEL_TYPE["TYPE"] == "YOLOv4"
                                else self.epochs-cfg.TRAIN["EVAL_EPOCHS"] )  ####

        self.train_dataset = data.Build_Dataset(
            anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"]
        )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN["BATCH_SIZE"],
            num_workers=cfg.TRAIN["NUMBER_WORKERS"],
            shuffle=True,
            #pin_memory=True,
        )

        self.yolov4 = Build_Model(self.weight_path, resume=resume, showatt=self.showatt).to(self.device)

        self.optimizer = optim.SGD(
            self.yolov4.parameters(),
            lr=cfg.TRAIN["LR_INIT"],
            momentum=cfg.TRAIN["MOMENTUM"],
            weight_decay=cfg.TRAIN["WEIGHT_DECAY"],
        )

        self.criterion = YoloV4Loss(
            anchors=cfg.MODEL["ANCHORS"],
            strides=cfg.MODEL["STRIDES"],
            iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"],
        )
        self.scheduler = CosineDecayLR(
            self.optimizer,
            T_max=self.epochs * len(self.train_dataloader),
            lr_init=cfg.TRAIN["LR_INIT"],
            lr_min=cfg.TRAIN["LR_END"],
            warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader),
        )
        if resume:
            self.__load_resume_weights()

    def __load_resume_weights(self):
        # default path: cfg.WEIGHT_PATH/last.pth
        # last_weight = osp.join(cfg.WEIGHT_PATH, "last.pth")
        chkpt = torch.load(self.weight_path, map_location=self.device)
        self.yolov4.load_state_dict(chkpt["model"])

        self.start_epoch = chkpt["epoch"] + 1
        if chkpt["optimizer"] is not None:
            self.optimizer.load_state_dict(chkpt["optimizer"])
            self.best_mAP = chkpt["best_mAP"]
        del chkpt
        self.logger.info_both("Load pretrained model from: %s"%self.weight_path)

    def __save_model_weights(self, epoch, mAP=-1):
        # if epoch>=epoch[last] or (epoch+1)%20==0[backup] or map>best_map[best] then save
        if epoch<self.eval_epoch and (epoch+1)%10!=0 and mAP<=self.best_mAP:
            return
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join( cfg.WEIGHT_PATH, "best.pth")
        last_weight = os.path.join( cfg.WEIGHT_PATH, "last.pth")
        chkpt = {
            "epoch":        epoch,
            "best_mAP":     self.best_mAP,
            "model":        self.yolov4.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
        }
        if epoch >= self.eval_epoch:
            torch.save(chkpt, last_weight)
            self.logger.info_both('Save model at: {:s}'.format(last_weight))

        if self.best_mAP == mAP:
            torch.save(chkpt["model"], best_weight)
            self.logger.info_both('Save model at: {:s}'.format(best_weight))

        if (epoch+1) % 10 == 0:
            torch.save(chkpt,  os.path.join(cfg.WEIGHT_PATH, "backup_epoch%d.pth" % (epoch+1)))
            self.logger.info_both('Save model: {:s}'.format("backup_epoch%d.pth" % (epoch+1)))
        del chkpt

    def train(self):
        self.logger.info_both(
            "Training start, img size is: {:d}, batchsize is: {:d}, work number is {:d}".format(
                cfg.TRAIN["TRAIN_IMG_SIZE"],
                cfg.TRAIN["BATCH_SIZE"],
                cfg.TRAIN["NUMBER_WORKERS"]
            )
        )
        # self.logger.info(self.yolov4)
        self.logger.info_both(
            "Train datasets number is : {:d}".format(len(self.train_dataset))
        )

        if self.fp_16:
            self.yolov4, self.optimizer = amp.initialize(
                self.yolov4, self.optimizer, opt_level="O1", verbosity=0
            )

        #--- train
        self.logger.info_both("      =======  Start  Training  ======     ")
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.yolov4.train()

            mloss = torch.zeros(4)
            self.logger.info_both("Epoch:[{:d}/{:d}]:".format(epoch+1, self.epochs))
            len_data = len(self.train_dataloader)
            with tqdm(self.train_dataloader, total=len_data, ncols=120, smoothing=0.9) as t:
                i = -1
                for (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in t:
                    i += 1
                    self.scheduler.step( len_data * epoch + i)

                    imgs = imgs.to(self.device)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    sbboxes = sbboxes.to(self.device)
                    mbboxes = mbboxes.to(self.device)
                    lbboxes = lbboxes.to(self.device)

                    p, p_d = self.yolov4(imgs)
                    loss, loss_ciou, loss_conf, loss_cls = self.criterion(
                        p, p_d, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes
                    )

                    if is_valid_number(loss.item()):
                        if self.fp_16:
                            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            loss.backward()
                    # Accumulate gradient for x batches before optimizing
                    if i % self.accumulate == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Update running mean of tracked metrics
                    loss_items = torch.tensor([loss_ciou, loss_conf, loss_cls, loss])
                    mloss = (mloss * i + loss_items) / (i+1)

                    # Print batch results
                    if i % 20 == 0:
                        self.logger.info(
                            "==== Step:[{:3d}/{:3d}], img_size:[{:3}], total_loss:{:.4f}|loss_ciou:{:.4f}|loss_conf:{:.4f}|loss_cls:{:.4f}|lr:{:.4f}".format(
                                i+1, len_data, self.train_dataset.img_size, mloss[3], mloss[0],
                                mloss[1], mloss[2], self.optimizer.param_groups[0]["lr"]
                            )
                        )
                    if i % 10 == 0:
                        self.writer.add_scalar("loss_ciou",  mloss[0], len_data*epoch + i)
                        self.writer.add_scalar("loss_conf",  mloss[1], len_data*epoch + i)
                        self.writer.add_scalar("loss_cls",   mloss[2], len_data*epoch + i)
                        self.writer.add_scalar("total_loss", mloss[3], len_data*epoch + i)

                    # multi-sclae training (320-608 pixels) every 10 batches
                    if self.multi_scale_train and (i + 1) % 10 == 0:
                        self.train_dataset.img_size = (
                            random.choice(range(10, 20)) * 32
                        )

                    # set tqdm post
                    t.set_postfix(  img_size    = self.train_dataset.img_size,
                                    loss_total  = mloss[3].item(),
                                    ciou        = mloss[0].item(),
                                    conf        = mloss[1].item(),
                                    cls         = mloss[2].item())

            # eval
            mAP = 0
            self.yolov4.eval()
            if epoch >= self.eval_epoch:
                self.logger.info_both("val img size is {}".format(cfg.VAL["TEST_IMG_SIZE"]))
                with torch.no_grad():
                    APs, inference_time = Evaluator(self.yolov4).calc_APs()
                    for cls in APs.keys():
                        self.logger.info_both("AP of class: {} = {}".format(cls, APs[cls]))
                        mAP += APs[cls]
                    mAP = mAP / self.train_dataset.num_classes
                    self.logger.info_both("Test mAP : {:.3f}".format(mAP))
                    self.logger.info_both("Best mAP : {:.3f}".format(self.best_mAP))
                    self.logger.info_both("Inference time: {:.2f} ms".format(inference_time))
                    self.writer.add_scalar("mAP", mAP, epoch)

            # save
            self.__save_model_weights(epoch, mAP)

            end = time.time()
            self.logger.info_both("==== cost time:{:.4f}s".format(end - start))
        self.logger.info_both( "End of training, best_test_mAP:{:.3f}%====".format( self.best_mAP ))
        #--- end of train

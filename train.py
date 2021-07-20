from tensorboardX import SummaryWriter
import argparse
from utils.logger import Logger
from utils.trainer import Trainer
from config.yolov4_config import PROJECT_PATH
import os.path as osp

if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_file",    type=str,   default="mobilenetv3.pth",
                                            help="weight file path, relative to WEIGHT_PATH")
    parser.add_argument("--resume",         type=bool,  default=False,      help="resume training flag")
    parser.add_argument("--gpu_id",         type=int,   default=0,          help="weather use GPU(0) or CPU(-1)" )
    parser.add_argument("--log_path",       type=str,   default="logs/",    help="log path, relative to PROJ_DIR")
    parser.add_argument("--accumulate",     type=int,   default=2,          help="batches to accumulate before optimizing")
    parser.add_argument("--fp_16",          type=bool,  default=False,      help="weather to use fp16 precision" )
    args = parser.parse_args()

    log_dir = osp.join(PROJECT_PATH, args.log_path)
    writer = SummaryWriter(logdir   =   osp.join(log_dir, "event") )
    logger = Logger(    log_path    =   osp.join(log_dir, "run.log"),
                        logger_name =   "YOLOv4"  )
    print("1")

    Trainer(logger      =   logger,
            writer      =   writer,
            weight_file =   args.weight_file,
            resume      =   args.resume,
            gpu_id      =   args.gpu_id,
            accumulate  =   args.accumulate,
            fp_16       =   args.fp_16  ).train()
"""
Generate train.txt and test.txt in DATASET_PATH
"""
import numpy as np
import os
import os.path as osp
import sys
Base_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(Base_dir, ".."))
import config.yolov4_config as cfg
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    split_percent = 0.75
    Data_path   = cfg.DATASET_PATH
    Img_path    = osp.join(Data_path, 'JPEGImages')
    train_index_file = osp.join(Data_path, 'train.txt')
    test_index_file = osp.join(Data_path, 'test.txt')

    file_list = []
    for file in os.listdir(Img_path):
        file_name, file_type = file.split('.')
        file_list.append(file_name)
            # shutil.move(
            #     file_path,
            #     osp.join(Data_path, 'Annotations')
            # )

    index = np.random.permutation(len(file_list))
    split_index = int(len(index) * split_percent)

    train_list  = np.array(file_list)[ index[:split_index] ]
    test_list   = np.array(file_list)[ index[split_index:] ]

    with open(train_index_file, "a") as f:
        for image_id in tqdm(train_list):
            f.write(image_id + "\n")

    with open(test_index_file, "a") as f:
        for image_id in tqdm(test_list):
            f.write(image_id + "\n")

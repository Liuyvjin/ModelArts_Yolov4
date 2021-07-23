# YOLOv4-pytorch for ModelArts

This is a pytorch implementation of YOLOv4 based on [argusswift/YOLOv4-pytorch](https://github.com/argusswift/YOLOv4-pytorch). You can train model on your own dataset and deploy it to the ModelArts easily.

## 1. Environment
* Windows
* python 3.6

Run the installation script to install all the dependencies.
```bash
pip install -r requirements.txt
```

## 2. Preparations
### 2.1 Dataset

This project supports datasets in Pascal VOC format. You need to place your data as follows:
```
ModelArts_Yolov4
├───data
│   └───Your_dataset
│       ├───Annotations
│       │   ├──1.xml
│       │   └──...
│       ├───JPEGImages
│       │   ├──1.jpg
│       │   └──...
```

Then:
* Update the `"DATASET_NAME"` and `"Customer_DATA"`in the `config/yolov4_config.py`.
Convert data format :use data/voc.py to *.txt

* Split the data into trainset and testset with `data/gen_img_index_file.py`. After that you will get two files: `train.txt` and `test.txt` in your dataset folder.

* Convert the pascal voc *.xml format annotation to *.txt format (Image_path &nbsp; xmin0,ymin0,xmax0,ymax0,class0 &nbsp;) using `data/convert_voc_to_txt.py`. You will get `data/train_annotation.txt`
and `data/test_annotation.txt`

* Generate annotation files for each class with `data/gen_cls_anno.py`. These files are generated in the `data/your_dataset/ClassAnnos/` directory and are used to calculate APs.

* Run `utils/anchor_kmeans.py`, which performs kmeans algrithom on the ground truth bboxes to get the most general anchor boxes. Update the `"MODEL['ANCHORS']"` in the `config/yolov4_config.py`.

### 2.2 Download Weight File
* Mobilenetv3 pre-trained weight:  [mobilenetv3](https://pan.baidu.com/s/1bfysmFMcpawWPe1KL1J4oA)(code: yolo)
* Make dir `weights/` in the ModelArts_Yolov4 and put the weight file in it.


# 3. Training

Run the following command to start training and see the details in the config/yolov4_config.py.
```
python -u train.py
```

During training, backups of model will be saved in `weights/*.pth`. You can interrupt training and resume training from these backups at any time using the following command.

```
python -u train.py --weight_file  your_backup.pth  --resume
```

# 5. Detection

Run `predict.py` and you can predict images from testset one by one.

![test1](https://github.com/Liuyvjin/ModelArts_Yolov4/data/test_images/test1.jpg)
![test2](https://github.com/Liuyvjin/ModelArts_Yolov4/data/test_images/test2.jpg)

# 4. Deployment

Copy `weights/best.pth` to `ModelArts/model/best.pth`. Then upload the entire `ModelArts` folder to the ModelArts platform.



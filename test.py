# created by Huang Lu
# 28/08/2016 13:44:54   
import os
from PIL import Image
import keras
import numpy as np
import random

import tensorflow as tf
from utils import visualize
from utils.config import Config
from utils.anchors import get_anchors
from utils.utils import mold_inputs,unmold_detections
from nets.mrcnn import get_train_model,get_predict_model
from nets.mrcnn_training import data_generator,load_image_gt
from dataset import ShapesDataset

from nets.resnet import get_resnet
# from nets.mobilenet import mobilenet_v2
# from nets.resnet50 import resnet50
# from nets.vgg16 import vgg16
# from nets.shufflecent_modify import shufflenet_v2_x1_0

# from  nets.mobilelenetv2_modify import  modelilev2_modify
from nets.mrcnn import get_train_model,get_predict_model


class ShapesConfig(Config):
    NAME = "shapes"
    GPU_COUNT = 1
    # 应该通过设置IMAGES_PER_GPU来设置BATCH的大小，而不是下面的BATCH_SIZE
    # BATCHS_SIZE自动设置为IMAGES_PER_GPU*GPU_COUNT
    # 请各位注意哈！
    IMAGES_PER_GPU = 1
    # BATCH_SIZE = 10
    NUM_CLASSES = 1 + 40
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # 训练集和验证集长度已经自动计算

if __name__ == "__main__":
    # model = mobilenet_v2(num_classes=4, pretrained=False).train()
    # model = resnet50(num_classes=4, pretrained=False).train()
    config = ShapesConfig()
    # model = get_train_model(config)
    model =get_predict_model(config)
    # model = modelilev2_modify(num_classes=1000, pretrained=False).train()
    model.summary()

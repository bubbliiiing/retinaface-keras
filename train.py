import keras
import numpy as np
from keras.optimizers import Adam
from nets.retinaface import RetinaFace
from nets.retinanet_training import Generator
from nets.retinanet_training import conf_loss, box_smooth_l1, ldm_smooth_l1
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from utils.utils import BBoxUtility
from utils.anchors import Anchors
from utils.config import cfg_re50, cfg_mnet


if __name__ == "__main__":
    #-------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    #-------------------------------#
    backbone = "mobilenet"
    training_dataset_path = './data/widerface/train/label.txt'

    if backbone == "mobilenet":
        cfg = cfg_mnet
        freeze_layers = 81
    elif backbone == "resnet50":  
        cfg = cfg_re50
        freeze_layers = 173
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    img_dim = cfg['image_size']

    #-------------------------------#
    #   创立模型
    #   请注意主干网络与预训练权重的
    #   对应
    #-------------------------------#
    model = RetinaFace(cfg, backbone=backbone)
    model_path = "model_data/retinaface_mobilenet025.h5"
    model.load_weights(model_path,by_name=True,skip_mismatch=True)

    #-------------------------------#
    #   获得先验框和工具箱
    #-------------------------------#
    anchors = Anchors(cfg, image_size=(img_dim, img_dim)).get_anchors()
    bbox_util = BBoxUtility(anchors)

    # 训练参数设置
    logging = TensorBoard(log_dir="logs")
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
        monitor='loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

    for i in range(freeze_layers): model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        Init_epoch = 0
        Freeze_epoch = 25
        # batch_size大小，每次喂入多少数据
        batch_size = 8
        # 最大学习率
        learning_rate_base = 1e-3

        gen = Generator(training_dataset_path,img_dim,batch_size,bbox_util)

        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen.generate(), 
                steps_per_epoch=gen.get_len()//batch_size,
                verbose=1,
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        Freeze_epoch = 25
        Epoch = 50
        # batch_size大小，每次喂入多少数据
        batch_size = 4
        # 最大学习率
        learning_rate_base = 1e-4

        gen = Generator(training_dataset_path,img_dim,batch_size,bbox_util)
        
        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen.generate(), 
                steps_per_epoch=gen.get_len()//batch_size,
                verbose=1,
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

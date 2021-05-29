import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from nets.retinaface import RetinaFace
from nets.retinaface_training import (ExponentDecayScheduler, Generator,
                                      LossHistory, box_smooth_l1, conf_loss,
                                      ldm_smooth_l1)
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import BBoxUtility

if __name__ == "__main__":
    #--------------------------------#
    #   获得训练用的人脸标签与坐标
    #--------------------------------#
    training_dataset_path = './data/widerface/train/label.txt'
    #-------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    #-------------------------------#
    backbone = "mobilenet"

    if backbone == "mobilenet":
        cfg = cfg_mnet
        freeze_layers = 81
    elif backbone == "resnet50":  
        cfg = cfg_re50
        freeze_layers = 173
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    img_dim = cfg['train_image_size']
    #--------------------------------------#
    #   载入模型与权值
    #   请注意主干网络与预训练权重的对应
    #--------------------------------------#
    model = RetinaFace(cfg, backbone=backbone)
    model_path = "model_data/retinaface_mobilenet025.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------#
    #   获得先验框和工具箱
    #-------------------------------#
    anchors = Anchors(cfg, image_size=(img_dim, img_dim)).get_anchors()
    bbox_util = BBoxUtility(anchors)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging         = TensorBoard(log_dir="logs/")
    checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5', monitor='loss', save_weights_only=True, save_best_only=False, period=1)
    reduce_lr       = ExponentDecayScheduler(decay_rate=0.92, verbose=1)
    early_stopping  = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)
    loss_history    = LossHistory("logs/")

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
        batch_size          = 8
        Init_epoch          = 0
        Freeze_epoch        = 50
        learning_rate_base  = 1e-3

        gen = Generator(training_dataset_path, img_dim, batch_size, bbox_util)

        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(weights=cfg['loc_weight']),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen, 
                steps_per_epoch=gen.get_len()//batch_size,
                verbose=1,
                epochs=Freeze_epoch,
                initial_epoch=Init_epoch,
                # 开启多线程可以加快数据读取的速度。
                # workers=4,
                # use_multiprocessing=True,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

    for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        batch_size          = 4
        Freeze_epoch        = 50
        Epoch               = 100
        learning_rate_base  = 1e-4

        gen = Generator(training_dataset_path, img_dim, batch_size, bbox_util)
        
        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(weights=cfg['loc_weight']),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(gen, 
                steps_per_epoch=gen.get_len()//batch_size,
                verbose=1,
                epochs=Epoch,
                initial_epoch=Freeze_epoch,
                # workers=4,
                # use_multiprocessing=True,
                callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history])

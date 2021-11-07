import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from nets.retinaface import RetinaFace
from nets.retinaface_training import box_smooth_l1, conf_loss, ldm_smooth_l1
from utils.anchors import Anchors
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.config import cfg_mnet, cfg_re50
from utils.dataloader import Generator
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
    #----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path为主干网络的权值，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path = "model_data/retinaface_mobilenet025.h5"
    #-------------------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    #-------------------------------------------------------------------#
    Freeze_Train = True
    #-------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，1代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #-------------------------------------------------------------------#
    num_workers = 1

    if backbone == "mobilenet":
        cfg             = cfg_mnet
        freeze_layers   = 81
    elif backbone == "resnet50":  
        cfg             = cfg_re50
        freeze_layers   = 173
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    model       = RetinaFace(cfg, backbone=backbone)
    if model_path != '':
        #------------------------------------------------------#
        #   载入预训练权重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------#
    #   获得先验框和工具箱
    #-------------------------------#
    anchors     = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()
    bbox_util   = BBoxUtility(anchors)

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

    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    #---------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #---------------------------------------------------------#
    #---------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #---------------------------------------------------------#
    if True:
        #----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        #----------------------------------------------------#
        batch_size          = 8
        Init_epoch          = 0
        Freeze_epoch        = 50
        learning_rate_base  = 1e-3

        gen = Generator(training_dataset_path, cfg['train_image_size'], batch_size, bbox_util)

        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(weights = cfg['loc_weight']),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(
            generator           = gen, 
            steps_per_epoch     = gen.get_len() // batch_size,
            epochs              = Freeze_epoch,
            initial_epoch       = Init_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

    if Freeze_Train:
        for i in range(freeze_layers): model.layers[i].trainable = True

    if True:
        #----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        #----------------------------------------------------#
        batch_size          = 4
        Freeze_epoch        = 50
        Epoch               = 100
        learning_rate_base  = 1e-4

        gen = Generator(training_dataset_path, cfg['train_image_size'], batch_size, bbox_util)
        
        model.compile(loss={
                    'bbox_reg'  : box_smooth_l1(weights=cfg['loc_weight']),
                    'cls'       : conf_loss(),
                    'ldm_reg'   : ldm_smooth_l1()
                },optimizer=keras.optimizers.Adam(lr=learning_rate_base)
        )

        model.fit_generator(
            generator           = gen, 
            steps_per_epoch     = gen.get_len()//batch_size,
            epochs              = Epoch,
            initial_epoch       = Freeze_epoch,
            use_multiprocessing = True if num_workers > 1 else False,
            workers             = num_workers,
            callbacks           = [logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )

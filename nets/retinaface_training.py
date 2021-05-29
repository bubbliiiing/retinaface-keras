import math
import os
import random
from random import shuffle

import cv2
import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image


def softmax_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-7)
    softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
    return softmax_loss

def conf_loss(neg_pos_ratio = 7,negatives_for_hard = 100):
    def _conf_loss(y_true, y_pred):
        #-------------------------------#
        #   取出先验框的数量
        #-------------------------------#
        num_boxes = tf.to_float(tf.shape(y_true)[1])
        
        labels         = y_true[:, :, :-1]
        classification = y_pred
        # --------------------------------------------- #
        #   分类的loss
        # --------------------------------------------- #
        cls_loss = softmax_loss(labels, classification)
        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   batch_size,
        # --------------------------------------------- #
        num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1)

        pos_conf_loss = tf.reduce_sum(cls_loss * y_true[:, :, -1], axis=1)
        # --------------------------------------------- #
        #   每一张图的负样本的个数
        #   batch_size,
        # --------------------------------------------- #
        num_neg = tf.minimum(neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat(axis=0, values=[num_neg, [(1 - has_min) * negatives_for_hard]])

        # --------------------------------------------- #
        #   从这里往后，与视频中看到的代码有些许不同。
        #   由于以前的负样本选取方式存在一些问题，
        #   我对该部分代码进行重构。
        #   求整个batch应该的负样本数量总和
        # --------------------------------------------- #
        num_neg_batch = tf.reduce_sum(tf.boolean_mask(num_neg, tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)

        # --------------------------------------------- #
        #   batch_size,8732
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = tf.reduce_sum(y_pred[:, :, 1:], axis=2)
        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs = tf.reshape(max_confs * (1 - y_true[:, :, -1]), [-1])
        _, indices = tf.nn.top_k(max_confs, k=num_neg_batch)

        neg_conf_loss = tf.gather(tf.reshape(cls_loss, [-1]), indices)

        # 进行归一化
        num_pos     = tf.where(tf.not_equal(num_pos, 0), num_pos, tf.ones_like(num_pos))
        total_loss  = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_loss /= tf.reduce_sum(num_pos)
        return total_loss
    return _conf_loss


def box_smooth_l1(sigma=1, weights=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #------------------------------------#
        #   取出作为正样本的先验框
        #------------------------------------#
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        #------------------------------------#
        #   计算 smooth L1 loss
        #------------------------------------#
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss * weights

    return _smooth_l1

def ldm_smooth_l1(sigma=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        #------------------------------------#
        #   取出作为正样本的先验框
        #------------------------------------#
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        #------------------------------------#
        #   计算 smooth L1 loss
        #------------------------------------#
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data(image, targes, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
    iw, ih = image.size
    h, w = input_shape
    box = targes

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    PRE_SCALES = [3.33, 2.22, 1.67, 1.25, 1.0]
    scale = random.choice(PRE_SCALES)
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # 将图像多余的部分加上灰条
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转图像
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 色域扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue*360
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:,:, 0]>360, 0] = 360
    x[:, :, 1:][x[:, :, 1:]>1] = 1
    x[x<0] = 0
    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255 # numpy array, 0 to 1

    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2,4,6,8,10,12]] = box[:, [0,2,4,6,8,10,12]]*nw/iw + dx
        box[:, [1,3,5,7,9,11,13]] = box[:, [1,3,5,7,9,11,13]]*nh/ih + dy
        if flip: 
            box[:, [0,2,4,6,8,10,12]] = w - box[:, [2,0,6,4,8,12,10]]
            box[:, [5,7,9,11,13]]     = box[:, [7,5,9,13,11]]
        
        center_x = (box[:, 0] + box[:, 2])/2
        center_y = (box[:, 1] + box[:, 3])/2
    
        box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

        box[:, 0:14][box[:, 0:14]<0] = 0
        box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
        box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h
        
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

    box[:, 4:-1][box[:,-1]==-1]=0
    box[:, [0,2,4,6,8,10,12]] /= w
    box[:, [1,3,5,7,9,11,13]] /= h
    box_data = box
    return image_data, box_data

class Generator(keras.utils.Sequence):
    def __init__(self, txt_path, img_size, batch_size, bbox_util):
        self.img_size = img_size
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.imgs_path, self.words = self.process_labels()
        self.bbox_util = bbox_util

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.imgs_path) / float(self.batch_size))

    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt','images/') + path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        return imgs_path, words

    def get_len(self):
        return len(self.imgs_path)
    
    def on_epoch_end(self):
        shuffle_index = np.arange(len(self.imgs_path))
        shuffle(shuffle_index)
        self.imgs_path = np.array(self.imgs_path)[shuffle_index]
        self.words = np.array(self.words)[shuffle_index]
        
    def __getitem__(self, index):
        inputs = []
        target0 = []
        target1 = []
        target2 = []
        
        for i in range(index*self.batch_size, (index+1)*self.batch_size):  
            img = Image.open(self.imgs_path[i])
            labels = self.words[i]
            annotations = np.zeros((0, 15))
            
            for idx, label in enumerate(labels):
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1
                annotations = np.append(annotations, annotation, axis=0)

            target = np.array(annotations)
            img, target = get_random_data(img, target, [self.img_size,self.img_size])

            # 计算真实框对应的先验框，与这个先验框应当有的预测结果
            assignment = self.bbox_util.assign_boxes(target)

            regression = assignment[:,:5]
            classification = assignment[:,5:8]

            landms = assignment[:,8:]
            
            inputs.append(img)     
            target0.append(np.reshape(regression,[-1,5]))
            target1.append(np.reshape(classification,[-1,3]))
            target2.append(np.reshape(landms,[-1,10+1]))
            if len(target0) == self.batch_size:
                tmp_inp = np.array(inputs)
                tmp_targets = [np.array(target0,dtype=np.float32),np.array(target1,dtype=np.float32),np.array(target2,dtype=np.float32)]
                
                inputs = []
                target0 = []
                target1 = []
                target2 = []
                return preprocess_input(tmp_inp), tmp_targets

class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self, decay_rate, verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose

    def on_epoch_end(self, batch, logs=None):
        lr = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % lr)

class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        import datetime
        curr_time       = datetime.datetime.now()
        time_str        = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))  
        self.losses     = []
        
        os.makedirs(self.save_path)

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

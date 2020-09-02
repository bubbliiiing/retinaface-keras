
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
from utils import backend
from PIL import Image
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2

def softmax_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-7)
    softmax_loss = -tf.reduce_sum(y_true * tf.log(y_pred),
                                    axis=-1)
    return softmax_loss

def conf_loss(neg_pos_ratio = 7,negatives_for_hard = 100):
    def _conf_loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        num_boxes = tf.to_float(tf.shape(y_true)[1])
        
        labels         = y_true[:, :, :-1]
        classification = y_pred

        cls_loss = softmax_loss(labels, classification)
        
        num_pos = tf.reduce_sum(y_true[:, :, -1], axis=-1)
        
        pos_conf_loss = tf.reduce_sum(cls_loss * y_true[:, :, -1],
                                      axis=1)
        # 获取一定的负样本
        num_neg = tf.minimum(neg_pos_ratio * num_pos,
                             num_boxes - num_pos)


        # 找到了哪些值是大于0的
        pos_num_neg_mask = tf.greater(num_neg, 0)
        # 获得一个1.0
        has_min = tf.to_float(tf.reduce_any(pos_num_neg_mask))
        num_neg = tf.concat( axis=0,values=[num_neg,
                                [(1 - has_min) * negatives_for_hard]])

        # 求平均每个图片要取多少个负样本
        num_neg_batch = tf.reduce_mean(tf.boolean_mask(num_neg,
                                                      tf.greater(num_neg, 0)))
        num_neg_batch = tf.to_int32(num_neg_batch)

        max_confs = y_pred[:, :, 1]

        # 取top_k个置信度，作为负样本
        x, indices = tf.nn.top_k(max_confs * (1 - y_true[:, :, -1]),
                                 k=num_neg_batch)

        # 找到其在1维上的索引
        batch_idx = tf.expand_dims(tf.range(0, batch_size), 1)
        batch_idx = tf.tile(batch_idx, (1, num_neg_batch))
        full_indices = (tf.reshape(batch_idx, [-1]) * tf.to_int32(num_boxes) +
                        tf.reshape(indices, [-1]))

        neg_conf_loss = tf.gather(tf.reshape(cls_loss, [-1]),
                                  full_indices)
        neg_conf_loss = tf.reshape(neg_conf_loss,
                                   [batch_size, num_neg_batch])
        neg_conf_loss = tf.reduce_sum(neg_conf_loss, axis=1)


        num_pos = tf.where(tf.not_equal(num_pos, 0), num_pos,
                            tf.ones_like(num_pos))
        total_loss = tf.reduce_sum(pos_conf_loss) + tf.reduce_sum(neg_conf_loss)
        total_loss /= tf.reduce_sum(num_pos)
        # total_loss = tf.Print(total_loss,[labels,full_indices,tf.reduce_sum(pos_conf_loss)/tf.reduce_sum(num_pos),tf.reduce_sum(neg_conf_loss)/tf.reduce_sum(num_pos),tf.reduce_sum(num_pos)])
        return total_loss
    return _conf_loss
def box_smooth_l1(sigma=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1

def ldm_smooth_l1(sigma=1):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
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

def get_random_data(image, targes, input_shape, random=True, jitter=.3, hue=.1, sat=1.5, val=1.5):
    iw, ih = image.size
    h, w = input_shape
    box = targes

    # 对图像进行缩放并且进行长和宽的扭曲
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(0.5,1.5)
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
        box[:, 0:14][box[:, 0:14]<0] = 0
        box[:, [0,2,4,6,8,10,12]][box[:, [0,2,4,6,8,10,12]]>w] = w
        box[:, [1,3,5,7,9,11,13]][box[:, [1,3,5,7,9,11,13]]>h] = h
        
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

    box[:,4:-1][box[:,-1]==-1]=0
    box[:, [0,2,4,6,8,10,12]] /= w
    box[:, [1,3,5,7,9,11,13]] /= h
    box_data = box
    return image_data, box_data

class Generator(object):
    def __init__(self, txt_path, img_size, batch_size, bbox_util):
        self.img_size = img_size
        self.txt_path = txt_path
        self.batch_size = batch_size
        self.imgs_path, self.words = self.process_labels()
        self.bbox_util = bbox_util

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
    
    def generate(self):
        while True:
            shuffle_index = np.arange(len(self.imgs_path))
            shuffle(shuffle_index)
            self.imgs_path = np.array(self.imgs_path)[shuffle_index]
            self.words = np.array(self.words)[shuffle_index]
            inputs = []
            target0 = []
            target1 = []
            target2 = []
            for i, image_path in enumerate(self.imgs_path):  
                img = Image.open(image_path)
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
                    yield preprocess_input(tmp_inp), tmp_targets

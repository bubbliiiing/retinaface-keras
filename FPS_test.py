import os
import time

import keras
import numpy as np
import torch
import tqdm
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from PIL import Image, ImageDraw, ImageFont

from retinaface import Retinaface
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import BBoxUtility, letterbox_image, retinaface_correct_boxes

class FPS_Retinaface(Retinaface):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)

        scale = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]]
        scale_for_landmarks = [np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
                                            np.shape(image)[1], np.shape(image)[0]]
        #---------------------------------------------------------#
        #   letterbox_image可以给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(self.input_shape[0], self.input_shape[1])).get_anchors()
            
        photo = np.expand_dims(preprocess_input(image),0)
        preds = self.retinaface.predict(photo)
        results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

        if len(results)>0:
            results = np.array(results)
            #---------------------------------------------------------#
            #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
            #---------------------------------------------------------#
            if self.letterbox_image:
                results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
        
            results[:,:4] = results[:,:4]*scale
            results[:,5:] = results[:,5:]*scale_for_landmarks
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                preds = self.retinaface.predict(photo)
                results = self.bbox_util.detection_out(preds, self.anchors, confidence_threshold=self.confidence)

                if len(results)>0:
                    results = np.array(results)
                    #---------------------------------------------------------#
                    #   如果使用了letterbox_image的话，要把灰条的部分去除掉。
                    #---------------------------------------------------------#
                    if self.letterbox_image:
                        results = retinaface_correct_boxes(results, np.array([self.input_shape[0], self.input_shape[1]]), np.array([im_height, im_width]))
                    
                    results[:,:4] = results[:,:4]*scale
                    results[:,5:] = results[:,5:]*scale_for_landmarks
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

if __name__ == '__main__':
    retinaface = FPS_Retinaface()
    test_interval = 100
    img = Image.open('img/street.jpg')
    tact_time = retinaface.get_FPS(img, test_interval)
    print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

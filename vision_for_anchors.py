from itertools import product as product
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from utils.config import cfg_mnet
from utils.anchors import Anchors


def decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox):
    # 获得先验框的宽与高
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    # 获得先验框的中心点
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width * 0.1
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height * 0.1
    decode_bbox_center_y += prior_center_y
    
    # 真实框的宽与高的求取
    decode_bbox_width = np.exp(mbox_loc[:, 2] * 0.2)
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] * 0.2)
    decode_bbox_height *= prior_height

    # 获取真实框的左上角与右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    prior_width = np.expand_dims(prior_width,-1)
    prior_height = np.expand_dims(prior_height,-1)
    prior_center_x = np.expand_dims(prior_center_x,-1)
    prior_center_y = np.expand_dims(prior_center_y,-1)

    mbox_ldm = mbox_ldm.reshape([-1,5,2])
    decode_ldm = np.zeros_like(mbox_ldm)
    decode_ldm[:,:,0] = np.repeat(prior_width,5,axis=-1)*mbox_ldm[:,:,0]*0.1 + np.repeat(prior_center_x,5,axis=-1)
    decode_ldm[:,:,1] = np.repeat(prior_height,5,axis=-1)*mbox_ldm[:,:,1]*0.1 + np.repeat(prior_center_y,5,axis=-1)


    # 真实框的左上角与右下角进行堆叠
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                    decode_bbox_ymin[:, None],
                                    decode_bbox_xmax[:, None],
                                    decode_bbox_ymax[:, None],
                                    np.reshape(decode_ldm,[-1,10])), axis=-1)
    # # 防止超出0与1
    # decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox

cfg     = cfg_mnet
cfg_mnet['image_size'] = 640
img_dim = cfg_mnet['image_size']
anchors = Anchors(cfg, image_size=(img_dim, img_dim)).get_anchors()
anchors = anchors[-800:]*img_dim

center_x = (anchors[:,0]+anchors[:,2])/2
center_y = (anchors[:,1]+anchors[:,3])/2

fig = plt.figure()
ax = fig.add_subplot(121)
plt.ylim(-300,900)
plt.xlim(-300,900)
ax.invert_yaxis()  #y轴反向

plt.scatter(center_x,center_y)

box_widths = anchors[0:2,2]-anchors[0:2,0]
box_heights = anchors[0:2,3]-anchors[0:2,1]

for i in [0,1]:
    rect = plt.Rectangle([anchors[i, 0],anchors[i, 1]], box_widths[i],box_heights[i],color="r",fill=False)
    ax.add_patch(rect)

ax = fig.add_subplot(122)
plt.ylim(-300,900)
plt.xlim(-300,900)
ax.invert_yaxis()  #y轴反向

plt.scatter(center_x,center_y)

mbox_loc = np.random.randn(800,4)
mbox_ldm = np.random.randn(800,10)

decode_bbox = decode_boxes(mbox_loc, mbox_ldm, anchors)
box_widths = decode_bbox[0:2,2]-decode_bbox[0:2,0]
box_heights = decode_bbox[0:2,3]-decode_bbox[0:2,1]

for i in [0,1]:
    rect = plt.Rectangle([decode_bbox[i, 0],decode_bbox[i, 1]], box_widths[i],box_heights[i],color="r",fill=False)
    plt.scatter((decode_bbox[i,2]+decode_bbox[i,0])/2,(decode_bbox[i,3]+decode_bbox[i,1])/2,color="b")
    ax.add_patch(rect)

plt.show()

import numpy as np
from math import ceil
from itertools import product as product


class Anchors(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(Anchors, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # 每个网格点2个先验框，都是正方形
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
     
        anchors = np.reshape(anchors,[-1,4])

        output = np.zeros_like(anchors[:,:4])
        output[:,0] = anchors[:,0] - anchors[:,2]/2
        output[:,1] = anchors[:,1] - anchors[:,3]/2
        output[:,2] = anchors[:,0] + anchors[:,2]/2
        output[:,3] = anchors[:,1] + anchors[:,3]/2

        if self.clip:
            output = np.clip(output, 0, 1)
        return output

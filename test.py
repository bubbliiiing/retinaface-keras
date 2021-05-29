#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from nets.retinaface import RetinaFace
from utils.config import cfg_mnet, cfg_re50

if __name__ == '__main__':
    model = RetinaFace(cfg_mnet, backbone="mobilenet")
    model.summary()

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)

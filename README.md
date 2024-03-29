## Retinaface：人脸检测模型在Keras当中的实现
---

## 目录
1. [仓库更新 Top News](#仓库更新)
2. [性能情况 Performance](#性能情况)
3. [所需环境 Environment](#所需环境)
4. [文件下载 Download](#文件下载)
5. [预测步骤 How2predict](#预测步骤)
6. [训练步骤 How2train](#训练步骤)
7. [评估步骤 Eval](#评估步骤)
8. [参考资料 Reference](#Reference)

## Top News
**`2022-03`**:**进行了大幅度的更新，支持step、cos学习率下降法、支持adam、sgd优化器选择、支持学习率根据batch_size自适应调整。**  
BiliBili视频中的原仓库地址为：https://github.com/bubbliiiing/retinaface-keras/tree/bilibili

**`2020-09`**:**仓库创建，支持模型训练，大量的注释，多个主干的选择，多个可调整参数。**   

## 性能情况
| 训练数据集 | 权值文件名称 | 测试数据集 | 输入图片大小 | Easy | Medium | Hard |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| Widerface-Train | retinaface_mobilenet025.h5 | Widerface-Val | 1280x1280 | 88.94% | 86.76% | 73.83% |
| Widerface-Train | retinaface_resnet50.h5 | Widerface-Val | 1280x1280 | 94.69% | 93.08% | 84.31% | 

## 所需环境
tensorflow-gpu==1.13.1  
keras==2.1.5  

## 文件下载
训练所需的retinaface_resnet50.h5、resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5等文件可以在百度云下载。     
链接: https://pan.baidu.com/s/1iiIqjlrtpvMjh_s2RsjSag 提取码: dru9     

数据集可以在如下连接里下载。      
链接: https://pan.baidu.com/s/1bsgay9iMihPlAKE49aWNTA 提取码: bhee     

## 预测步骤
### a、使用预训练权重
1. 下载完库后解压，运行predict.py，输入  
```python
img/timg.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
_defaults = {
    "model_path"        : 'model_data/retinaface_mobilenet025.h5',
    "backbone"          : 'mobilenet',
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    #----------------------------------------------------------------------#
    #   是否需要进行图像大小限制。
    #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
    #   keras代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
    #   会导致检测结果偏差，主干为resnet50不存在此问题。
    #----------------------------------------------------------------------#
    "input_shape"       : [1280, 1280, 3],
    "letterbox_image"   : True
}

```
3. 运行predict.py，输入  
```python
img/timg.jpg
```  
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 训练步骤
1. 本文使用widerface数据集进行训练。  
2. 可通过上述百度网盘下载widerface数据集。  
3. 覆盖根目录下的data文件夹。  
4. 根据自己需要选择**从头开始训练还是在已经训练好的权重下训练**，需要修改train.py文件下的代码，在训练时需要**注意backbone和权重文件的对应**。
使用mobilenet为主干特征提取网络的示例如下：   
从头开始训练：    
```python
#-------------------------------#
#   创立模型
#-------------------------------#
model = RetinaFace(cfg, backbone=backbone)
model_path = "model_data/mobilenet_2_5_224_tf_no_top.h5"
model.load_weights(model_path,by_name=True,skip_mismatch=True)
```
在已经训练好的权重下训练：   
```python
#-------------------------------#
#   创立模型
#-------------------------------#
model = RetinaFace(cfg, backbone=backbone)
model_path = "model_data/retinaface_mobilenet025.h5"
model.load_weights(model_path,by_name=True,skip_mismatch=True)
```
5. 可以在logs文件夹里面获得训练好的权值文件。  

## 评估步骤  
1. 在retinaface.py文件里面，在如下部分修改model_path和backbone使其对应训练好的文件。  
```python
_defaults = {
    "model_path"        : 'model_data/retinaface_mobilenet025.h5',
    "backbone"          : 'mobilenet',
    "confidence"        : 0.5,
    "nms_iou"           : 0.45,
    #----------------------------------------------------------------------#
    #   是否需要进行图像大小限制。
    #   开启后，会将输入图像的大小限制为input_shape。否则使用原图进行预测。
    #   keras代码中主干为mobilenet时存在小bug，当输入图像的宽高不为32的倍数
    #   会导致检测结果偏差，主干为resnet50不存在此问题。
    #----------------------------------------------------------------------#
    "input_shape"       : [1280, 1280, 3],
    "letterbox_image"   : True
}

```
2. 下载好百度网盘上上传的数据集，其中包括了验证集，解压在根目录下。 
3. 运行evaluation.py即可开始评估。


## Reference
https://github.com/biubug6/Pytorch_Retinaface


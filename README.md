#### 人脸表情识别

##### 一、论文相关 

论文链接:[https://arxiv.org/abs/1710.07557](https://arxiv.org/abs/1710.07557)

###### 1.数据集

使用fer2013数据集，该数据集包含35887张人脸表情，，每张图片是由大小固定为48×48的灰度图像组成，共有7种表情，分别对应于数字标签0-6，即0: 'angry'(生气), 1: 'disgust'(恶心), 2: 'fear'(害怕), 3: 'happy'(高兴), 4: 'sad'(伤心), 5: 'surprise'(惊讶), 6: 'neutral'(不悲不喜)。该数据集中，没有具体的人脸图片，而是多个csv文件（包含人脸数据、用途和标签）。

可以用一些方式将csv文件转换为人脸灰度图，具体参考[人脸表情数据集-fer2013](https://blog.csdn.net/rookie_wei/article/details/83659595)

###### 2.模型

在传统的用于特征提取的CNN中，最后都会包含全连接层，这会导致模型中存在大量参数。为了减少参数，该论文对全连接层进行了修改：

1. 第一个模型(sequential fully-CNN)使用全局平均池来代替全连接层，具体来说就是在最后的卷积层中保持feature map数量和分类数目相同，并且将softmax激活函数用到每个feature map中。这个模型是一个标准的全卷积神经网络，由9个卷积层、ReLU、批量归一化和全局平均池组成，大约包含600000个参数。

2. 第二个模型参考了[Xception体系结构](https://arxiv.org/abs/1610.02357)。 这种架构结合了residual modules和depth-wise  separable  convolution。 residual modules会修改两个后续层之间的desired mapping，以使学习的特征成为original feature map和desired features之间的差值。因此，desired  features H(x)和 learning problem F(X)之间的关系为：$H(x) = F(x) + x$。depth-wise  separable  convolution由两个不同的层组成：depth-wise  convolutions和 point-wise  convolutions，这两层的主要作用是将空间自相关与通道自相关分离。具体而言就是，首先在每M个输入通道上应用D×D滤波器，然后再应用N 个1×1×M卷积滤波器，将M个输入通道组合为N个输出通道。 应用1×1×M卷积可将feature map中的每个值组合在一起，而无需考虑通道内的空间关系。depth-wise  separable  convolution相对于标准卷积减少了$\frac{1}{N}+\frac{1}{D^2}$倍的计算。

3. 最终的模型(mini-Xception)是一个完全卷积的神经网络，其中包含4个residual  depth-wise  separable  convolutions，每个卷积后面都进行了批量归一化操作和ReLU激活函数， 最后一层使用全局平均池和softmax激活函数来预测结果。 最终的模型具有大约60000个参数，和第一个模型相比减少了10倍，与原始CNN相比减少了80倍。模型结构如下：

   ![模型结构](https://tva1.sinaimg.cn/large/007S8ZIlgy1gh034r4u6ij30ct0hawfe.jpg)

##### 二、模型代码

###### 1.目录结构

![代码结构](https://tva1.sinaimg.cn/large/007S8ZIlgy1gh036iycasj308706v0t8.jpg)

images目录中存放待识别的图片(所有图片都从互联网上下载)，results目录中存放识别结果，trained_models中是预训练好的模型。utils目录下的dataset.py文件用于读取数据集和标签，inference.py文件用于人脸检测和图片加框加字，preprocessor.py文件用于数据预处理。image_emotion.py文件就是用来识别表情的文件。

###### 2.环境需求

```txt
python3.5
Package                 Version
----------------------- -------
cmake                   3.18.0
face-recognition        1.3.0
h5py                    2.7.0
imageio                 2.9.0
Keras                   2.0.5
matplotlib              3.0.3
numpy                   1.13.3
opencv-python           3.2.0.6
pandas                  0.19.1
Pillow                  7.2.0
pip                     20.2b1
scikit-image            0.15.0
scipy                   1.4.1
statistics              1.0.3.5
tensorflow              1.1.0
```

###### 3.使用

```txt
首先cd到src目录下，然后输入
python image_emotion.py ../images/image_001.jpg
这样就会在results目录下看到输出predicted_image_001.jpg
如果环境搭建失败，可以在我的个人服务器上试试：
ip：121.199.47.228
username：root
password：Root@123
用户名和密码会在7.31之后过期
首先cd cd zq_dir/face_emotion/src
然后workon emotion
最后python image_emotion.py ../images/image_001.jpg
```

###### 4.代码解读

首先加载预训练模型并获取数据标签，然后读取图片，并使用face_recognition库对图片进行人脸检测，接着将得到的人脸灰度图经过模型分类器进行预测分类，最后就会在results目录下输出预测图片。

github地址：[https://github.com/liyiersan/face_emotion](https://github.com/liyiersan/face_emotion)


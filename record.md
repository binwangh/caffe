# 来源https://github.com/hujie-frank/SENet
修改src/caffe/layers/pooling_layer.cu 发现原来的global average pooling效率比较低

添加了Axpy层，主要用于 channel-wise scale和element-wise summation，其中，官方的Eltwise已经具备了点乘的操作。这个应该是更进一步的封装。
使用方法如下：

# Imagenet classification with deep convolutional neural network
非饱和神经元（non-saturating neurons）、ReLU Nonlinearity，提出了ReLU = max(0, *)：使得训练速度更快

Local Response Normalization：

Overlapping Pooling

Data Augmentation

Dropout防止过拟合

# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
训练深度神经网络很复杂，因为前一层的参数发生变化，在训练过程中，每层的输入分布也会发生变化。通过较低的学习率和好的初始化参数来减慢训练，但使得训练非线性的模型非常困难。

通过使用Internal Covariate Shift来解决：允许使用更高的学习率并且不太关心初始化参数，并且在某些情况下消除了对Dropout的需要。

注意：BatchNormLayer + ScaleLayer 需要搭配使用。



#coding=utf-8
"""
2017年02月11日
通过pycaffe测试网络层的功能。
需要指定***.prototxt的链接，与Caffe类似，这样更加方便。

"""

import os
import numpy as np
import _init_paths
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2


def createNet(prototxt):
    net = caffe.Net(prototxt, caffe.TEST)
    return net

def forward(net, data1, data2):
    # net.blobs[***]: ***代表layer_name 此时先初始化data数据，之后得到处理后的数据。
    net.blobs['data1'].data[...] = data1
    net.blobs['data2'].data[...] = data2
    net.forward()
    return net.blobs['scale'].data

def caffeScaleTest(prototxt, data1, data2):
    net = createNet(prototxt)
    return forward(net, data1, data2)

if __name__ == '__main__':

    prototxt = 'temp.prototxt'
    x1 = np.random.rand(2, 3, 3)
    x2 = np.random.rand(2, 3, 3)
    y = caffeScaleTest(prototxt, x1, x2)
    print("x1:")
    print(x1)
    print("x2:")
    print(x2)
    print("y:")
    print(y)


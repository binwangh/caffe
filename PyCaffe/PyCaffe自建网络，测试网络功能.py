#coding=utf-8
"""
2017年02月11日
通过pycaffe测试网络层的功能。在这个脚本中创建网络结构（temp.prototxt），且不需要使用Model。

"""
import _init_paths
import os
import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def createNet(shape, coeff):
    blobShape = caffe_pb2.BlobShape()
    blobShape.dim.extend(shape)
    n = caffe.NetSpec()
    # n.***：***代表name   L.***：***代表具体的Layers层
    n.data1 = L.Input(shape=[blobShape])
    n.data2 = L.Input(shape=[blobShape])
    n.eltwise = L.Eltwise(n.data1, n.data2, operation=P.Eltwise.SUM)

    prototxt = 'temp.prototxt'
    with open(prototxt, 'w') as f:
        f.write(str(n.to_proto()))
    net = caffe.Net(prototxt, caffe.TEST)
    #os.remove(prototxt)
    return net

def forward(net, data):
    net.blobs['data1'].data[...] = data
    net.blobs['data2'].data[...] = data
    net.forward()
    return net.blobs['eltwise'].data

def caffeEltwiseTest(data, coeff):
    net = createNet(data.shape, coeff)
    return forward(net, data)

if __name__ == '__main__':
    x = np.random.rand(1, 2, 3)
    y = caffeEltwiseTest(x, coeff=[1.0])
#    print(np.allclose(y, x))
    print(x)
    print(y)

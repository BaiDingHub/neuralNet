#!/usr/bin/env python
# coding: utf-8

# # 加载数据

# In[1]:


import pickle
import numpy as np
import struct

# In[2]:


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


# In[3]:


def loadCifarData():
    '''
    输出
    X_train:(50000,32,32,3)
    Y_train_Y:(50000,)
    X_test_X:(10000,32,32,3)
    Y_test_Y:(10000,)
    
    '''
    filename = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    train_data = load_file('cifar-10-batches-py/data_batch_1')
    X_train = train_data['data']
    Y_train = train_data['labels']
    for i in range(1,len(filename)):
        batch = load_file('cifar-10-batches-py/'+filename[i])
        X_train = np.vstack((X_train,batch['data']))
        Y_train = np.hstack((Y_train,batch['labels']))
    X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    test_data = load_file('cifar-10-batches-py/test_batch')
    X_test = test_data['data'].reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y_test = np.array(test_data['labels']).reshape((10000,))
    return X_train,Y_train,X_test,Y_test


def loadMnistImg(filename):
    '''
    输入
    filename: Mnist图片所在的路径
    输出
    imgs：(60000,784)
    '''
    binfile = open(filename, 'rb') # 读取二进制文件
    buffers = binfile.read()

    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组

    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum = head[1]
    width = head[2]
    height = head[3]

    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'

    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组

    binfile.close()
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组

    return imgs


def loadMnistLabel(filename):
    '''
    输入
    filename: Mnist's Label所在的路径
    输出
    labels:(60000,)
    '''
    binfile = open(filename, 'rb') # 读二进制文件
    buffers = binfile.read()
 
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数
 
    labelNum = head[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置
 
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据
 
    binfile.close()
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)
 
    return labels


def loadMnistData():
    '''
    输出
    trainX:(60000,784)
    trainY:(60000,)
    testX:(10000,784)
    testY:(10000,)
    
    '''
    trainImg= 'mnist/train-images.idx3-ubyte'
    trainLabel= 'mnist/train-labels.idx1-ubyte'

    testImg= 'mnist/t10k-images.idx3-ubyte'
    testLabel= 'mnist/t10k-labels.idx1-ubyte'

    trainX = loadMnistImg(trainImg)
    trainY = loadMnistLabel(trainLabel)
    testX = loadMnistImg(testImg)
    testY = loadMnistLabel(testLabel)
    return trainX,trainY,testX,testY
#!/usr/bin/env python
# coding: utf-8

# # 加载数据

# In[1]:


import pickle
import numpy as np


# In[2]:


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data


# In[3]:


def load_data():
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



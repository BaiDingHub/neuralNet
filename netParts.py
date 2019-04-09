#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# ## FC

# In[ ]:


def affine_forward(x,w,b):
    '''
    前项传播，H个神经元
    输入：
    x：(N，D1,D2,D3...)
    w:(D,H)
    b:(H,)
    输出:
    out:运算结果,(N,H)
    cache:包括，(x,w,b)
    '''
    x_c = x.reshape((x.shape[0],-1))
    out = np.dot(x_c,w) + b
    cache = (x,w,b)
    return out,cache


# In[ ]:


def affine_backward(dout,cache):
    """
    反向传播,H个神经元
    输入：
    dout：该层out的倒数值，（N,H)
    cace：该层的（x,w,b)
    x：(N，D1,D2,D3...)
    w:(D,H)
    b:(H,)
    输出：
    dx，dw,db
    dx：(N，D1,D2,D3...)
    dw:(D,H)
    db:(H,)
    """
    x,w,b = cache
    s = x.shape
    z = x.reshape((x.shape[0],-1))
    dx = np.dot(dout,w.T).reshape(s)
    dw = np.dot(z.T,dout)
    db = np.sum(dout,axis=0)
    return dx,dw,db


# ## conv

# In[14]:


def conv_forward_naive(x,w,b,conv_param):
    '''
    input:
    -x:(N,H,W,C)
    -w:(F,HH,WW,C)
    -b:(F,)
    conv_param:一个字典,包含'stride','pad'
    
    output:
    -out:(N,Hout,Wout,F)  其中
        Hout = 1+(H + 2*pad - HH)/stride
        Wout = 1+(W + 2*pad - WW)/stride
    -cache:(x,w,b,conv_param)
    
    '''
    N,H,W,C = x.shape
    F,HH,WW,_ = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    Hout = int(1+(H + 2*pad - HH)/stride)
    Wout = int(1+(W + 2*pad - WW)/stride)
    out = np.zeros((N,Hout,Wout,F))   
    
    pad_x = np.lib.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),mode = 'constant',
                             constant_values = 0)   #填充X   

    for n in range(N):
        for f in range(F):
            out[n,:,:,f] = np.ones((Hout,Wout)) * b[f]
            for i in range(Hout):
                for j in range(Wout):
                    out[n,i,j,f] += np.sum(pad_x[n,i*stride:i*stride + HH,j*stride:j*stride + WW,:] * w[f])
    cache = (x,w,b,conv_param)
    return out,cache


# In[15]:


def conv_backward_naive(dout,cache):
    '''
    input:
    -dout:(N,Hout,Wout,F)
    -cache:包含(x,w,b,conv_param)
    
    return:
    -dx:(N,H,W,C)
    -dw:(F,HH,WW,C)
    -db:(F,)
    '''
    x,w,b,conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N,Hout,Wout,F = dout.shape
    N,H,W,C = x.shape
    F,HH,WW,_ = w.shape
    pad_x = np.lib.pad(x,((0,0),(pad,pad),(pad,pad),(0,0)),mode = 'constant',
                             constant_values = 0)   #填充X
    pad_dx = np.zeros_like(pad_x)
        
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    for n in range(N):
        for f in range(F):
            for i in range(Hout):
                for j in range(Wout):
                    dw[f] += pad_x[n,i*stride:i*stride + HH,j*stride:j*stride + WW,:] * dout[n,i,j,f]
                    db[f] += dout[n,i,j,f]
                    pad_dx[n,i*stride:i*stride + HH,j*stride:j*stride + WW,:] += w[f] * dout[n,i,j,f]
    dx = pad_dx[:,pad:pad+H,pad:pad+W,:]
    return dx,dw,db


# ## pooling

# In[22]:


def max_pool_forward_naive(x,pool_param):
    '''
    Input:
    -x:输入数据,(N,H,W,C)
    -pool_param:一个字典,包含
        -'pool_h':池化层的高度
        -'pool_w':池化层的宽度
        -'stride':步长
        
    Output:
    -out:输出数据,(N,Hout,Wout,C)
    -cache:(x,pool_param)
    '''
    N,H,W,C = x.shape
    pool_h = pool_param['pool_h']
    pool_w = pool_param['pool_w']
    stride = pool_param['stride']
    Hout = int(1+(H  - pool_h)/stride)
    Wout = int(1+(W  - pool_w)/stride)
    
    out = np.zeros((N,Hout,Wout,C))
    
    for n in range(N):
        for c in range(C):
             for i in range(Hout):
                    for j in range(Wout):
                        out[n,i,j,c] = np.max(x[n,i*stride:i*stride + pool_h,j*stride:j*stride+pool_w,c])
    cache = (x,pool_param)
    return out,cache


# In[23]:


def max_pool_backward_naive(dout,cache):
    '''
    Input:
    -dout:梯度,(N,Hout,Wout,C)
    -cache:
        -x:(N,H,W,C)
        -pool_param:一个字典
            -'pool_h':池化层的高度
            -'pool_w':池化层的宽度
            -'stride':步长
    Output:
    -dx:(N,H,W,C)
    '''
    x,pool_param = cache
    N,H,W,C = x.shape
    N,Hout,Wout,C = dout.shape
    pool_h = pool_param['pool_h']
    pool_w = pool_param['pool_w']
    stride = pool_param['stride']
    
    dx = np.zeros_like(x)
    for n in range(N):
        for c in range(C):
            for i in range(Hout):
                for j in range(Wout):
                    window = x[n,i*stride:i*stride + pool_h,j*stride:j*stride+pool_w,c]
                    dx[n,i*stride:i*stride + pool_h,j*stride:j*stride+pool_w,c] = (window == np.max(window)) * dout[n,i,j,c]
    return dx


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 神经网络模型
# 目前有二层神经网络，多层神经网络

# In[1]:


import numpy as np
from netParts import *
from optim import *


# In[ ]:


def TwoLayerNet(x,y,H,L,lr,iterations):
    '''
    两层神经网络
    H，隐藏层的神经元
    L,输出层的神经元
    lr:学习率
    iterations:display代次数
    
    '''
    N = x.reshape(x.shape[0],-1).shape[1]
    w1 = np.random.randn(N,H)
    b1 = np.zeros((H,))
    w2 = np.random.randn(H,L)
    b2 = np.zeros((L,))
    for i in range(iterations):
        out1,cache1 = affine_forward(x,w1,b1)
        out2,cache2 = affine_forward(out1,w2,b2)
        loss,dout = svm_loss(out2,y)
        dout1,dw2,db2 = affine_backward(dout,cache2)
        dx,dw1,db1 = affine_backward(dout1,cache1)
        print(loss)
        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1
        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2


# In[ ]:


class FullyConnectedNets(object):
    '''
    全连接神经网络
    '''
    def __init__(self,input_dims,hidden_dims,num_classes,
                 loss_function = svm_loss,activation_function = 'relu',
                weight_scale = 1e-2,reg = 0,
                 use_norm = None,dropout = 1,
                 eps = 1e-8,bn_momentum = 0.9,
                config = {'lr':1e-5},grad_function = sgd):
        '''
        Inputs:
        -input_dims:数据特征的维数
        -hidden_dims:一个list，里面包含各个隐藏层的神经元数目
        -num_classes:预测标签的数目
        -loss_function:loss函数,有svm_loss,softmax_loss
        -activation_function:激活函数，有sigmoid,relu,tanh
        -reg:正则化参数,默认0
        -norm:批标准化，None为不标准化,默认None
        -dropout:dropout操作，为消除神经元的百分比，1表示不dropout,默认1
        -weight_scale:初始化W矩阵的权重
        -eps:批标准化的参数
        -bn_momentum:批标准化的更新权重
         -config:梯度下降的参数
            -lr:学习率，默认1e-5
            -momentum:momentum和adam的参数,默认0.9
            -decay_rate:rmsprop和adam的参数,默认0.99
            -eps:rmsprop和adam的参数，默认1e-8
        -grad_function:梯度下降函数,默认sgd
            -sgd
            -rmsprop
            -adam
            -momentum
        
        参数:
        -self.params:一个字典，包含了 每一层的 信息，用 W1,b1等表示
        '''
        self.depth = len(hidden_dims) + 1
        self.loss_function = loss_function
        self.reg = reg
        self.use_norm = use_norm
        self.use_dropout = dropout != 1
        self.dropout = dropout
        self.activation_forward = None
        self.activation_backward = None
        self.bn_params = [{'mode':'train','eps':eps,'momentum':bn_momentum} for i in range(self.depth)]
        self.dp_params = {'mode':'train','dropout':self.dropout}
        self.config = config
        self.grad_function = grad_function
        
        
        
        ## 初始化参数
        self.params = {}
        D = input_dims
        for i,H in enumerate(hidden_dims):
            self.params['W%d'%(i+1)] = weight_scale * np.random.randn(D,H)
            self.params['b%d'%(i+1)] = np.zeros((H,))
            if self.use_norm:                    #如果使用批标准化
                self.params['gamma%d'%(i+1)] = np.ones((H,))
                self.params['beta%d'%(i+1)] = np.zeros((H,))
                self.bn_params[i]['running_mean'] = np.zeros((H,))
                self.bn_params[i]['running_var'] = np.zeros((H,))
            D = H
        self.params['W%d'%(i+2)] = weight_scale * np.random.randn(D,num_classes)
        self.params['b%d'%(i+2)] = np.zeros((num_classes,))
        if self.use_norm:
                self.params['gamma%d'%(i+2)] = np.ones((num_classes,))
                self.params['beta%d'%(i+2)] = np.zeros((num_classes,))
                self.bn_params[i+1]['running_mean'] = np.zeros((num_classes,))
                self.bn_params[i+1]['running_var'] = np.zeros((num_classes,))
        
        
        ## 设置激活函数
        if activation_function == 'relu':
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward
        if activation_function == 'sigmoid':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward
        if activation_function == 'tanh':
            self.activation_forward = tanh_forward
            self.activation_backward = tanh_backward
        
    def loss(self,x,y):
        '''
        Inputs:
        -x：数据
        -y:标签
        '''
        grads = {}
        AllCache = {}
        AllOut = {}
        for i in range(self.depth):                        #forward
            #全连接层
            out,cache = affine_forward(x,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)])
            AllOut["affineOut%d"%(i+1)] = out
            AllCache['affineCache%d'%(i+1)] = cache
            #批标准化层
            if self.use_norm:
                out,cache = norm_forward(out,self.params['gamma%d'%(i+1)],self.params['beta%d'%(i+1)],self.bn_params[i])
                AllOut['normOut%d'%(i+1)] = out
                AllCache['normCache%d'%(i+1)] = cache
            #dropout
            if self.use_dropout:
                out,cache = dropout_forward(out,self.dp_params)
                AllOut['dropOut%d'%(i+1)] = out
                AllCache['dropCache%d'%(i+1)] = cache
            #激活函数
            out,cache = self.activation_forward(out)
            AllOut['activationOut%d'%(i+1)] = out
            AllCache['activationCache%d'%(i+1)] = cache
            x = out
        loss,dout = self.loss_function(x,y)                #computer loss
        for i in reversed(range(self.depth)):              #backward
            #激活函数
            dx = self.activation_backward(dout,AllCache['activationCache%d'%(i+1)])
            #dropout
            if self.use_dropout:
                dx = dropout_backward(dx,AllCache['dropCache%d'%(i+1)])
            #批标准化层
            if self.use_norm:
                dx,dgamma,dbeta = norm_backward(dx,AllCache['normCache%d'%(i+1)])
                grads['gamma%d'%(i+1)] = dgamma
                grads['beta%d'%(i+1)] = dbeta
            #全连接层
            dx,dw,db = affine_backward(dx,AllCache['affineCache%d'%(i+1)])
            
            grads['W%d'%(i+1)] = dw + self.reg * self.params['W%d'%(i+1)]
            grads['b%d'%(i+1)] = db
            dout = dx
            loss += 0.5 * self.reg * np.sqrt(np.sum(self.params['W%d'%(i+1)] ** 2))  #正则化
        return loss,grads
    
    def predict(self,x,y = None):
        '''
        得到score和acc，
        若y为None，只返回score
        '''
        for i in range(self.depth):              #forward
            out,cache = affine_forward(x,self.params['W%d'%(i+1)],self.params['b%d'%(i+1)])
            if self.use_norm:
                if y.all() == None:
                    self.bn_params[i]['mode'] = 'test'
                out,cache = norm_forward(out,self.params['gamma%d'%(i+1)],self.params['beta%d'%(i+1)],self.bn_params[i])
            if self.use_dropout:
                if y.all() == None:
                    self.dp_params['mode'] = 'test'
                out,cache = dropout_forward(out,self.dp_params)
            out,cache = self.activation_forward(out)
            x = out
        score =np.argmax(x,axis=1)
        if y.all() == None:
            return score,_
        acc = np.sum(score == y) / y.shape[0]
        return score,acc


# In[ ]:


class ConvNets(object):
    '''
    卷积神经网络
    '''
    def __init__(self,input_dims,conv_dims,pool_dims,fc_dims,num_classes,
                loss_function = svm_loss,activation_function = 'relu',
                 pool_function = 'max_pool',
                weight_scale = 1e-2,reg = 0,
                 use_norm = None,dropout = 1,
                 eps = 1e-5,bn_momentum = 0.9,
                config = {'lr':1e-5},grad_function = sgd):
        '''
        Inputs:
        -input_dims:数据特征(H,W,C)
        -conv_dims:一个list，元素为元组，
            -(h,w,f,stride,pad) 一个隐藏层卷积核的宽度，高度，数目,步长，填充
        -pool_dims:一个list,元素为元组,
            -(h,w,stride) 一个隐藏层池化层的宽度，高度，步长
        -fc_dims:一个list,元素为int，每一个fc连接层神经元的个数
        -num_classes:预测标签的数目
        -loss_function:loss函数,有svm_loss,softmax_loss
        -activation_function:激活函数，有sigmoid,relu,tanh
        -pool_function:池化层的类型
        -reg:正则化参数,默认0
        -norm:批标准化，None为不标准化,默认None
        -dropout:dropout操作，为消除神经元的百分比，1表示不dropout,默认1
        -weight_scale:初始化W矩阵的权重
        -eps:批标准化的参数
        -bn_momentum:批标准化的更新权重
        -config:梯度下降的参数
            -lr:学习率，默认1e-5
            -momentum:momentum和adam的参数,默认0.9
            -decay_rate:rmsprop和adam的参数,默认0.99
            -eps:rmsprop和adam的参数，默认1e-8
        -grad_function:梯度下降函数,默认sgd
            -sgd
            -rmsprop
            -adam
            -momentum
        
        参数:
        -self.params:一个字典，包含了 每一层的 信息，用 W1,b1等表示
        '''
        self.conv_depth = len(conv_dims)
        self.fc_depth = len(fc_dims)+ 1
        self.loss_function = loss_function
        self.reg = reg
        self.use_norm = use_norm
        self.use_dropout = dropout != 1
        self.dropout = dropout
        self.activation_forward = None
        self.activation_backward = None
        self.conv_bn_params = [{'mode':'train','eps':eps,'momentum':bn_momentum} for i in range(self.conv_depth)]
        self.fc_bn_params = [{'mode':'train','eps':eps,'momentum':bn_momentum} for i in range(self.fc_depth)]
        self.dp_params = {'mode':'train','dropout':self.dropout}
        self.config = config
        self.grad_function = grad_function
        
        ## 初始化参数
        self.params = {}      #所有的参数，包括卷积层，池化层，全连接层
        Ho,Wo,C = input_dims
        # initalize conv params
        for i,layer in enumerate(zip(conv_dims,pool_dims)):
            conv,pool = layer
            #conv
            H,W,F,stride,pad = conv
            self.params['conv_W%d'%(i+1)] = weight_scale * np.random.randn(F,H,W,C)
            self.params['conv_b%d'%(i+1)] = np.zeros((F,))
            self.params['conv_stride%d'%(i+1)] = stride
            self.params['conv_pad%d'%(i+1)] = pad
            #after conv
            Ho = int(1+(Ho + 2*pad - H)/stride)
            Wo = int(1+(Wo + 2*pad - W)/stride)
            #pooling
            H,W,stride = pool
            self.params['pool_H%d'%(i+1)] = H
            self.params['pool_W%d'%(i+1)] = W
            self.params['pool_stride%d'%(i+1)] = stride
            #after pooling
            Ho = int(1 + (Ho - H)/stride)
            Wo = int(1 + (Wo - W)/stride)
            #norm initialization
            if self.use_norm:                    
                self.params['conv_gamma%d'%(i+1)] = np.ones((C,))
                self.params['conv_beta%d'%(i+1)] = np.zeros((C,))
                self.conv_bn_params[i]['running_mean'] = np.zeros((C,))
                self.conv_bn_params[i]['running_var'] = np.zeros((C,))
            C = F
        #initalize fc params
        D = Ho * Wo * F   #fc 层的input dim
        for i,H in enumerate(fc_dims):
            self.params['fc_W%d'%(i+1)] = weight_scale * np.random.randn(D,H)
            self.params['fc_b%d'%(i+1)] = np.zeros((H,))
            if self.use_norm:                    #如果使用批标准化
                self.params['fc_gamma%d'%(i+1)] = np.ones((H,))
                self.params['fc_beta%d'%(i+1)] = np.zeros((H,))
                self.fc_bn_params[i]['running_mean'] = np.zeros((H,))
                self.fc_bn_params[i]['running_var'] = np.zeros((H,))
            D = H
        self.params['fc_W%d'%(i+2)] = weight_scale * np.random.randn(D,num_classes)
        self.params['fc_b%d'%(i+2)] = np.zeros((num_classes,))
        if self.use_norm:
                self.params['fc_gamma%d'%(i+2)] = np.ones((num_classes,))
                self.params['fc_beta%d'%(i+2)] = np.zeros((num_classes,))
                self.fc_bn_params[i+1]['running_mean'] = np.zeros((num_classes,))
                self.fc_bn_params[i+1]['running_var'] = np.zeros((num_classes,))
                     
        
        ## 设置激活函数
        if activation_function == 'relu':
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward
        if activation_function == 'sigmoid':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward
        if activation_function == 'tanh':
            self.activation_forward = tanh_forward
            self.activation_backward = tanh_backward
        
        #设置pool function
        if pool_function == 'max_pool':
            self.pool_forward = max_pool_forward_naive
            self.pool_backward = max_pool_backward_naive
        
        self.conv_forward = conv_forward_naive
        self.conv_backward = conv_backward_naive
            
    def loss(self,x,y):
        '''
        Inputs:
        -x：数据
        -y:标签
        '''
        grads = {}
        AllCache = {}
        AllOut = {}
        #forward
        #卷积层
        for i in range(self.conv_depth):
            #conv
            conv_param = {}
            conv_param['stride'] = self.params['conv_stride%d'%(i+1)]
            conv_param['pad'] = self.params['conv_pad%d'%(i+1)]
            out,cache = self.conv_forward(x,self.params['conv_W%d'%(i+1)],self.params['conv_b%d'%(i+1)],conv_param)
            AllOut["convOut%d"%(i+1)] = out
            AllCache['convCache%d'%(i+1)] = cache
            #标准化
            if self.use_norm:
                out,cache = spatial_batchnorm_forward(out,self.params['conv_gamma%d'%(i+1)],self.params['conv_beta%d'%(i+1)],self.conv_bn_params[i])
                AllOut['conv_normOut%d'%(i+1)] = out
                AllCache['conv_normCache%d'%(i+1)] = cache
            #激活函数
            out,cache = self.activation_forward(out)
            AllOut['activationConvOut%d'%(i+1)] = out
            AllCache['activationConvCache%d'%(i+1)] = cache
            #pooling
            pool_param = {}
            pool_param['pool_h'] = self.params['pool_H%d'%(i+1)]
            pool_param['pool_w'] = self.params['pool_W%d'%(i+1)]
            pool_param['stride'] = self.params['pool_stride%d'%(i+1)]
            out,cache = self.pool_forward(out,pool_param)
            AllOut["poolOut%d"%(i+1)] = out
            AllCache['poolCache%d'%(i+1)] = cache
            
            x = out
        #全连接层
        for i in range(self.fc_depth):
            #fc
            out,cache = affine_forward(x,self.params['fc_W%d'%(i+1)],self.params['fc_b%d'%(i+1)])
            AllOut["affineOut%d"%(i+1)] = out
            AllCache['affineCache%d'%(i+1)] = cache
            #批标准化层
            if self.use_norm:
                out,cache = norm_forward(out,self.params['fc_gamma%d'%(i+1)],self.params['fc_beta%d'%(i+1)],self.fc_bn_params[i])
                AllOut['fc_normOut%d'%(i+1)] = out
                AllCache['fc_normCache%d'%(i+1)] = cache
            #dropout
            if self.use_dropout:
                out,cache = dropout_forward(out,self.dp_params)
                AllOut['dropOut%d'%(i+1)] = out
                AllCache['dropCache%d'%(i+1)] = cache
            #激活函数
            out,cache = self.activation_forward(out)
            AllOut['activationFcOut%d'%(i+1)] = out
            AllCache['activationFcCache%d'%(i+1)] = cache
            
            x = out
        #compute loss
        loss,dout = self.loss_function(x,y)
        #fc_backward
        for i in reversed(range(self.fc_depth)):         
            #激活函数
            dx = self.activation_backward(dout,AllCache['activationFcCache%d'%(i+1)])
            #dropout
            if self.use_dropout:
                dx = dropout_backward(dx,AllCache['dropCache%d'%(i+1)])
            #批标准化层
            if self.use_norm:
                dx,dgamma,dbeta = norm_backward(dx,AllCache['fc_normCache%d'%(i+1)])
                grads['fc_gamma%d'%(i+1)] = dgamma
                grads['fc_beta%d'%(i+1)] = dbeta
            #全连接层
            dx,dw,db = affine_backward(dx,AllCache['affineCache%d'%(i+1)])
            
            grads['fc_W%d'%(i+1)] = dw + self.reg * self.params['fc_W%d'%(i+1)]
            grads['fc_b%d'%(i+1)] = db
            dout = dx
            loss += 0.5 * self.reg * np.sqrt(np.sum(self.params['fc_W%d'%(i+1)] ** 2))  #正则化
        #conv_backward
        for i in reversed(range(self.conv_depth)):
            #pooling
            dx = self.pool_backward(dout,AllCache['poolCache%d'%(i+1)])
            #激活函数
            dx = self.activation_backward(dx,AllCache['activationConvCache%d'%(i+1)])
            #标准化
            if self.use_norm:
                dx,dgamma,dbeta = spatial_batchnorm_backward(dx,AllCache['conv_normCache%d'%(i+1)])
                grads['conv_gamma%d'%(i+1)] = dgamma
                grads['conv_beta%d'%(i+1)] = dbeta
            #conv
            dx,dw,db = self.conv_backward(dx,AllCache['convCache%d'%(i+1)])
            
            grads['conv_W%d'%(i+1)] = dw + self.reg * self.params['conv_W%d'%(i+1)]
            grads['conv_b%d'%(i+1)] = db
            dout = dx
            loss += 0.5 * self.reg * np.sqrt(np.sum(self.params['conv_W%d'%(i+1)] ** 2))  #正则化
        return loss,grads
    
    def predict(self,x,y = None):
        '''
        得到score和acc，
        若y为None，只返回score
        '''
        #conv_forward
        for i in range(self.conv_depth):
            #conv
            conv_param = {}
            conv_param['stride'] = self.params['conv_stride%d'%(i+1)]
            conv_param['pad'] = self.params['conv_pad%d'%(i+1)]
            out,cache = self.conv_forward(x,self.params['conv_W%d'%(i+1)],self.params['conv_b%d'%(i+1)],conv_param)
            if self.use_norm:
                if y.all() == None:
                    self.conv_bn_params[i]['mode'] = 'test'
                out,cache = spatial_batchnorm_forward(out,self.params['conv_gamma%d'%(i+1)],self.params['conv_beta%d'%(i+1)],self.conv_bn_params[i])
            #激活函数
            out,cache = self.activation_forward(out)

            #pooling
            pool_param = {}
            pool_param['pool_h'] = self.params['pool_pH%d'%(i+1)]
            pool_param['pool_w'] = self.params['pool_pW%d'%(i+1)]
            pool_param['stride'] = self.params['pool_pstride%d'%(i+1)]
            out,cache = self.pool_forward(out,pool_param)
            
            x = out
            
        #fc_forward
        for i in range(self.fc_depth):              #forward
            out,cache = affine_forward(x,self.params['fc_W%d'%(i+1)],self.params['fc_b%d'%(i+1)])
            if self.use_norm:
                if y.all() == None:
                    self.fc_bn_params[i]['mode'] = 'test'
                out,cache = norm_forward(out,self.params['fc_gamma%d'%(i+1)],self.params['fc_beta%d'%(i+1)],self.fc_bn_params[i])
            if self.use_dropout:
                if y.all() == None:
                    self.dp_params['mode'] = 'test'
                out,cache = dropout_forward(out,self.dp_params)
            out,cache = self.activation_forward(out)
            x = out
        score =np.argmax(x,axis=1)
        if y.all() == None:
            return score,_
        acc = np.sum(score == y) / y.shape[0]
        return score,acc


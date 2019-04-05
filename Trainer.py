#!/usr/bin/env python
# coding: utf-8

# # 神经网络模型训练器
# 包括训练，等

# In[ ]:


import numpy as np


# In[ ]:


class ModelProcessor(object):
    '''
    包含处理model的各种函数
    '''
    def __init__(self,model,x,y):
        '''
        input:
        model:神经网络模型
        x:输入数据
        y:输入标签
        

        self.reg:正则化参数
        self.dropout:dropout参数,表示保留神经元的百分比
        self.norm:批标准化参数,None为不标准化，
        '''

        self.reg = model.reg
        self.dropout = model.dropout
        self.use_norm = model.use_norm
        self.model = model
        self.x = x
        self.y = y
    
    def train(self,epoch = 5,iterations = 1000,printFreq = 2):
        '''
        训练神经网络模型
        输入:
        epoch:训练整个数据集的次数,默认5
        iterations:每次训练数据的个数,默认1000
        printFreq:每五次迭代输出loss
        '''
        loss_history = []
        acc_history = []
        iter_epoch_num = (int(self.x.shape[0] / iterations) + 1)    #一个迭代次数
        iter_nums = iter_epoch_num * epoch
        configs = {}
        for i in range(iter_nums):
            random_index = np.random.choice(self.x.shape[0],iterations)
            batch_x = self.x[random_index]
            batch_y = self.y[random_index]
            loss ,grads= self.model.loss(batch_x,batch_y)
            score,acc = self.model.predict(batch_x,batch_y)
            loss_history.append(loss)
            acc_history.append(acc)
            for value in grads.keys():                  #更新参数
                if i == 0:
                    configs[value] = self.model.config.copy()
                self.model.params[value] = self.model.grad_function(self.model.params[value],grads[value],configs[value])    
            if (i+1) % printFreq == 0:                  #输出loss
                print('第',int(i/iter_epoch_num)+1,'/',epoch,'epoch','第',i %iter_epoch_num + 1,'次迭代:',
                      'loss:',loss,'||    accuracy:',acc)
        return loss_history,acc_history


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # 该文件对各个模块进行应用
# 

# In[1]:


import numpy as np 
from load_datas import *
import matplotlib.pyplot as plt
import neuralNet as net
import netParts as parts
import Trainer 


# ## 数据加载

# In[2]:


X_train,Y_train,X_test,Y_test = load_data()     #加载数据


# In[ ]:


classes = [x for x in range(10)]
for c in classes:
    images = X_train[Y_train == c][:7]
    for index,image in enumerate(images):
        plt.subplot(7,10,index*10 + c+1)
        plt.imshow(image.astype('uint8'))
        plt.axis('off')

plt.show()


# ## 两层神经网络

# In[ ]:


x = X_train[:100]
y = Y_train[:100]


# In[ ]:


# net.TwoLayerNet(x,y,100,10,1e-4,100)            #两层神经网络


# ## 多层神经网络

# In[8]:


model = net.FullyConnectedNets(3*32*32,[500,100],10,lr=1e-8,reg=0.6,
                               loss_function=parts.softmax_loss,activation_function = 'relu',use_norm = True)   #多层神经网络


# In[9]:


x = X_train[:3000]
y = Y_train[:3000]


# In[11]:


#loss,grads = model.loss(x,y)


# In[12]:


processor = Trainer.ModelProcessor(model,x,y)               #训练器


# In[13]:


loss_history,acc_history = processor.train(epoch=10,iterations=150,printFreq=15)


# In[ ]:


test_x = X_test[:100]
test_y = Y_test[:100]


# In[ ]:


score,acc = model.predict(test_x,test_y)
print(acc)


# In[ ]:


plt.plot(loss_history)
plt.title('loss')
plt.xlabel('itertions')
plt.ylabel('loss')
plt.show()


# In[ ]:


plt.plot(acc_history)
plt.show()


# ## 其他

# In[ ]:





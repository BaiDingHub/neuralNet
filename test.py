#!/usr/bin/env python
# coding: utf-8

# # 该文件对各个模块进行应用
# 

# In[ ]:


import numpy as np 
from load_datas import *
import matplotlib.pyplot as plt
import neuralNet as net
import netParts as parts
import optim as op
import Trainer 


# ## 数据加载

# In[ ]:


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


net.TwoLayerNet(x,y,100,10,1e-4,100)            #两层神经网络


# ## 多层神经网络

# In[ ]:


model = net.FullyConnectedNets(3*32*32,[500,100],10,
                               loss_function=op.svm_loss,activation_function = 'relu',
                              config = {'lr':1e-5,'momentum':0.9,'decay_rate':0.99},grad_function = net.rmsprop)   #多层神经网络


# In[ ]:


x = X_train[:3000]
y = Y_train[:3000]


# In[ ]:


#loss,grads = model.loss(x,y)


# In[ ]:


#print(loss)


# In[ ]:


processor = Trainer.ModelProcessor(model,x,y)               #训练器


# In[ ]:


loss_history,acc_history = processor.train(epoch=10,iterations=50,printFreq=20)


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


# ## 卷积神经网络

# In[ ]:


conv_dims = [(3,3,5,2,2),(3,3,3,2,2)]
pool_dims = [(3,3,1),(3,3,1)]
fc_dims = [500,100]
model = net.ConvNets((32,32,3),conv_dims,pool_dims,fc_dims,10, config = {'lr':1e-5,'momentum':0.9,'decay_rate':0.99},reg=0.6)


# In[ ]:


x = X_train[:3000]
y = Y_train[:3000]


# In[ ]:


loss,grads = model.loss(x,y)


# In[ ]:


print(loss)


# In[ ]:


processor = Trainer.ModelProcessor(model,x,y)               #训练器


# In[ ]:


loss_history,acc_history = processor.train(epoch=2,iterations=1000,printFreq=1)


# In[ ]:


plt.plot(loss_history)
plt.title('loss')
plt.xlabel('itertions')
plt.ylabel('loss')
plt.show()


# In[ ]:


plt.plot(acc_history)
plt.show()


# In[ ]:





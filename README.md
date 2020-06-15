# neuralNet
numpy来写神经网络和CNN
该神经网络使用cifar数据集和MNIST数据集

**注意**：

在运行前，请先下载CIFAR或者MNIST数据集，将其解压后放置在对应的文件夹下面。


# 该篇使用numpy来写神经网络模型，共完成了以下内容：
## 1.普通神经网络： 
      前向传播
      反向传播
      且层数可变
## 2.各种优化策略
      1）各种loss函数
          1.softmax损失函数
          2.svm损失函数
          
      2）各种激活函数
          1.relu
          2.sigmoid
          3.tanh
          
      3）batchnormalizaiton
      
      4）dropout
      
      5）spatial_batchnorm （ 空间标准化，用于CNN
      
      6）各种梯度下降方法

## 3.CNN模型
     conov层
     max_pool层

## 其他文件
   1） Train.py
        用于对模型进行整体的训练
   2）test.py
        用于对模型进行测试
      

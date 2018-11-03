'''
Created on 2018年11月2日

@author: zhouweixin
@note: 入口函数
'''
import torch
from load_data import load_data1, load_data2
from train import train

# 需要注意的是：当数据集没有进行归一化成范围较小的数时，
# 请使用较小的学习率，不然可能会不收敛, 小数据集更易于收敛

#=======================================================
#========================数据集1========================
#=======================================================

# 加载数据集
x, y = load_data1() # 数值较大的数据集
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

# train("build_model1", x, y, 50000, 3e-4)
train("build_model2", x, y, 50000, 3e-4)

#=======================================================
#========================数据集2========================
#=======================================================

# 加载数据集
x, y = load_data2() # 数值较小的数据集
x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

# train("build_model1", x, y, 5000, 1e-2)
train("build_model2", x, y, 5000, 1e-2)

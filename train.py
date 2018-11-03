'''
Created on 2018年11月3日

@author: zhouweixin
'''

import torch
import torch.nn as nn
import torch.optim as optim
import linear_model


def train(model_name, x, y, epoch_num=50000, lr=3e-4):
    # 创建模型
    model = getattr(linear_model, model_name)()
    # 创建loss
    criterion = nn.MSELoss()
    # 创建优化函数
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for i in range(epoch_num):
        # forward
        output = model(x)
        loss = criterion(output, y)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 1000 == 0:
            # 输入信息
            if model_name == 'build_model1':
                print('epoch = %d, loss = %.8f, y = %.4fx + %.4f' % ((i + 1), loss.item(), model.linear.weight.data.item(), model.linear.bias.data.item()))
            else:
                print('epoch = %d, loss = %.8f, y = %.4fx + %.4f' % ((i + 1), loss.item(), model.w.data.item(), model.b.data.item()))

'''
Created on 2019/2/27 13:58

@author : zhouweixin
@note   : 利用pytorch自定义模型及自定义数据集
'''

import torch
from torch import nn, optim
from torch.nn import Parameter
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LinearModel(nn.Module):
    """自定义模型(不用nn.Linear)
    y = 2x + 3
    """
    def __init__(self):
        super(LinearModel, self).__init__()
        self.w = Parameter(torch.Tensor([0.01]))
        self.b = Parameter(torch.zeros([1]))

    def forward(self, x):
        return self.w * x + self.b

class MyDataset(Dataset):
    """
    自定义数据集, 加载自己的数据, 需要重写__len__和__getitem__函数
    """
    def __init__(self, x, y):
        super(MyDataset, self).__init__()

        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.size(0)

def load_data():
    '''较小数据集：0~9.9
    '''

    x = []
    y = []
    for i in range(100):
        x.append(i * 0.1)
        y.append(2 * (i * 0.1) + 3)

    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y


# 加载数据
x, y = load_data()
train_data = MyDataset(x, y)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

epoch_num = 100
learning_rate = 0.01

# 创建模型
model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), learning_rate)

for epoch in range(epoch_num):
    total_loss = 0.
    for x, y in train_loader:
        # forward
        pred = model(x)
        # compute loss
        loss = criterion(pred, y)

        optimizer.zero_grad()
        # backward
        loss.backward()
        # update parameters
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    print('%d / %d, loss = %.6f' % (epoch+1, epoch_num, total_loss / len(train_loader.dataset)))

print('训练完成: w = %.2f, b = %.2f' % (model.w, model.b))
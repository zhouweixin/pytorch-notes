'''
Created on 2018年11月2日

@author: zhouweixin
@note: pytorch实现自定义线性模型：y = 2x + 3
            用pytorch的网络类创建自定义的参数，达到模型参数的效果，方便以后灵活自定义网络

'''

import torch
import torch.nn as nn

class LinearModel1(nn.Module):
    def __init__(self):
        super(LinearModel1, self).__init__()
        
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
class LinearModel2(nn.Module):
    def __init__(self):
        super(LinearModel2, self).__init__()
        
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return self.w * x + self.b
    
def build_model1():
    return LinearModel1()

def build_model2():
    return LinearModel2()

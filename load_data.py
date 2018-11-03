'''
Created on 2018年11月2日

@author: zhouweixin
@note: 创建0~99的数据集，数据集较大，需要用较小的学习率去学习
'''

import numpy as np

def load_data1():
    '''较大数据集：0~99
    需要用稍小点的学习率去训练
    '''
    x = []
    y = []
    for i in range(100):
        x.append(i + 0.1)
        y.append(2 * (i + 0.1) + 3)
    
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return x, y

def load_data2():
    '''较小数据集：0~9.9
    需要用稍大点的学习率去训练
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

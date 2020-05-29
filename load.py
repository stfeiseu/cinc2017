from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
import scipy.io as sio
import keras
from sklearn.preprocessing import scale

MAX_LEN = 18000

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)    # 如果输入的ecg为全部训练集，则返回的均值和方差为全部训练集的均值和方差
        self.classes = sorted(set(label for label in labels))   # 这个地方每个label应该是列表或数组
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):

        x = [(ecg - self.mean) / self.std for ecg in x]
        x = pad(x)
        #print('x shape is ' + str(x.shape))
        x = x[:, :, None]
        #print('x[:,:,None] shape is ' + str(x.shape))
        return x

    def process_y(self, y):

        y = [self.class_to_int[c] for c in y]
        y = keras.utils.np_utils.to_categorical(y, num_classes=len(self.classes))
        return y

def pad(x, val=0, dtype=np.float32):   # 将所有数据pad成相同长度，长度不够的在后边补0

    max_len = MAX_LEN
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.hstack(x)
    mean = np.mean(x).astype(np.float32)
    std = np.std(x).astype(np.float32)
    return (mean, std)





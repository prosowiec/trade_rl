import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def transform_training_set(training_set : np.array, train_split = .7):
    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    seq_length = 16
    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * train_split)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))
    
    dataset = {
        "dataX" : dataX,
        "dataY" : dataY,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY
    }
    
    
    
    return dataset

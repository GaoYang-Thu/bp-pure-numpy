# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:22:30 2018

Todo: achieve BP algorithm without using any deep learning library

@author: YangGao
"""
import numpy as np

from neural_network import NeuralNetwork # import the neural network class

import config as cf # import config parameters
x = cf.X
y = cf.Y
iter_num = cf.ITER_NUM

from loss_array import loss_array, plot_loss # import loss function related defs

if __name__ == '__main__':
    
    nn = NeuralNetwork(x,y)
    loss_arr = loss_array()
    
    for i in range(iter_num):
        nn.feedforward()
        nn.backprobagation()
        loss_arr[i] = 0.5 * np.dot((nn.y - nn.output).T, (nn.y - nn.output))
        
    print(nn.output)
    
    plot_loss(loss_arr)
    
        

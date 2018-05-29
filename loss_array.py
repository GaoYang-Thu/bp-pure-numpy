# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:55:34 2018

@author: YangGao
"""

import numpy as np
import config as cf
import matplotlib.pyplot as plt

def loss_array():
    loss_array = np.zeros((cf.ITER_NUM,1))
    return loss_array

def plot_loss(loss_array):
#    plt.ion()
    fig = plt.figure()
    plt.plot(range(cf.ITER_NUM),loss_array)
    plt.xlabel('iteration')
    plt.ylabel('Loss')
    plt.show()

    

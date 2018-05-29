# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:40:21 2018

@author: YangGao
"""
import numpy as np

ITER_NUM = 30000

X = np.array([[0,0,1],
              [1,1,0],
              [1,0,1],
              [1,1,1]])

Y = np.array([[0],[1],[1],[0]])

LEARNING_RATE = 0.02
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:31:36 2018

@author: YangGao
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))
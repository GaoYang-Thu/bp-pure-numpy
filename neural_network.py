# -*- coding: utf-8 -*-
"""
Created on Mon May 28 15:42:05 2018

Todo: create a neural network class

@author: YangGao
"""
import numpy as np
from sigmoid_def import sigmoid, sigmoid_d

class NeuralNetwork():
    def __init__(self, x,y):
        self.input =  x
        self.weights_1 = np.random.rand(self.input.shape[1],4)
        self.weights_2 = np.random.rand(4,1)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer_1 = sigmoid(np.dot(self.input, self.weights_1))
        self.output = sigmoid(np.dot(self.layer_1, self.weights_2))
        
    def backprobagation(self):
        d_weights_2 = np.dot(self.layer_1.T, 
                             (2*(self.y - self.output) * sigmoid_d(self.output))
                             );
        d_weights_1 = np.dot(self.input.T, 
                             (np.dot(2*(self.y - self.output) * sigmoid_d(self.output), 
                                     self.weights_2.T) * sigmoid_d(self.layer_1)
                                    )
                             )

        self.weights_1 += d_weights_1
        self.weights_2 += d_weights_2
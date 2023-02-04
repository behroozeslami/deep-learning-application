# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 17:10:01 2023

@author: eslamimossallamb
"""

'''

Implementation of DenseNet in pyTorch.
https://towardsdatascience.com/understanding-and-visualizing-densenets-7f688092391a
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html#DenseNet

'''

import torch.nn as nn
import torch

class DenseLayer(nn.Module):
    
    def __init__(self, c_in, bn_size, growth_rate):
        '''
        Inputs:
            c_in - Number of input channels
            bn_size - Bottleneck size (factor of growth rate) for the output of the 1x1 convolution. Typically between 2 and 4.
            growth_rate - Number of output channels of the 3x3 convolution
        '''
        super(DenseLayer,self).__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.cnn1 = nn.Conv2d(in_channels=c_in, out_channels=bn_size*growth_rate, kernel_size=1,stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.cnn2 = nn.Conv2d(in_channels=bn_size*growth_rate, out_channels=growth_rate, kernel_size=3,stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.cnn1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.cnn2(out)
        out = torch.cat([x,out], 1)
        
        return out
    
class TransitionLayer(nn.Module):
    
    def __init__(self, c_in, c_out):
        '''
        '''
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.cnn = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1,stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2,stride=2)
            
    def forward(self, x):
        
        out = self.bn(x)
        out = self.relu(out)
        out = self.cnn(out)
        out = self.pool(out)
        
        return out
        
class DenseBlock(nn.Module):

    def __init__(self, c_in, num_layers, bn_size, growth_rate):
        """
        Inputs:
            c_in - Number of input channels
            num_layers - Number of dense layers to apply in the block
            bn_size - Bottleneck size to use in the dense layers
            growth_rate - Growth rate to use in the dense layers
            act_fn - Activation function to use in the dense layers
        """
        super().__init__()
        
        layers = []
        for n in range(num_layers):
            layer = DenseLayer(c_in, bn_size, growth_rate)
            layers.append(layer)
            c_in += growth_rate
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        
        out = self.block(x)
        
        return out
    
class DenseNet(nn.Module):

    def __init__(self, num_classes=10, num_layers=[6,6,6,6], bn_size=2, growth_rate=16):
        super().__init__()
        
        c_hidden = growth_rate * bn_size
        
        self.input_net = nn.Conv2d(3, c_hidden, kernel_size=3, padding=1)
        
        blocks_list = []
        for block_idx, block_size in enumerate(num_layers):
            block = DenseBlock(c_hidden, block_size, bn_size, growth_rate)
            blocks_list.append(block)
            c_hidden += growth_rate*block_size
            if block_idx < len(num_layers)-1:        # Don't apply transition layer on last block
                blocks_list.append(TransitionLayer(c_hidden, c_hidden//2))
                c_hidden = c_hidden//2
        self.blocks = nn.Sequential(*blocks_list)
        
        self.output_net = nn.Sequential(nn.BatchNorm2d(c_hidden), # The features have not passed a non-linearity until here.
                                        nn.ReLU(),
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        nn.Linear(c_hidden, num_classes))

    def forward(self, x):
        
        out = self.input_net(x)
        out = self.blocks(out)
        out = self.output_net(out)
        
        return out
        
    def init_model_params(self):
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
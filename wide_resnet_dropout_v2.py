# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 12:31:30 2022

@author: behro
"""

import torch.nn as nn

def make_resnet_architecture(w=32, input_channels=3, num_classes=10, out_cnn=16, k=2, 
                             block_sizes = [2,2,2],Pdropout=0): 
    
    architecture = dict()
    architecture['input_channels'] = input_channels
    architecture['num_classes'] = num_classes
    architecture['out_cnn'] = out_cnn
    architecture['blocks'] = []
    in_block = out_cnn
    num_downsample = 0;
    for (i,bock_size) in enumerate(block_sizes):
        out_block = 2**(4+i)*k
        if in_block!=out_block:
            num_downsample += 1 
        extended_block = dict()
        extended_block['in'] = in_block
        extended_block['out'] = out_block
        extended_block['block_size'] = bock_size
        in_block = out_block
        architecture['blocks'].append(extended_block)
    w_end = w/2**num_downsample
    architecture['AvePool_kernel'] = int(w_end)
    architecture['Pdropout'] = Pdropout
    architecture['fc'] = out_block
    
    return architecture


class BasicBlock(nn.Module):
    
    def __init__(self,in_channels, out_channels, Pdropout=0):
        
        super(BasicBlock,self).__init__()
        
        down_sample = (in_channels!=out_channels)
        
        stride = 1;
        if down_sample:
            stride = 2
            
        cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=stride, padding=1)
        bn1 = nn.BatchNorm2d(in_channels)
        cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,stride=1, padding=1)
        bn2 = nn.BatchNorm2d(out_channels)
        dropout = nn.Dropout(Pdropout)
        self.relu = nn.ReLU()
        
        self.main_path = nn.Sequential(bn1, self.relu, cnn1, dropout, bn2, self.relu, cnn2)
        
        if down_sample:
            cnn3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=stride, padding=1)
            self.shortcut = nn.Sequential(cnn3)
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
        
        out_1 = self.main_path(x)
        out_2 = self.shortcut(x)
        out = out_1+out_2
        
        return out
        
class WideResnet(nn.Module):
    
    def __init__(self, architecture, block_class):
        super().__init__()
        self.architecture = architecture
        cnn = nn.Conv2d(in_channels=architecture['input_channels'], 
                        out_channels=architecture['out_cnn'], 
                        kernel_size=3,stride=1, padding=0)
        relu = nn.ReLU()
        bn = nn.BatchNorm2d(architecture['blocks'][-1]['out'])
        blocks = self.make_extended_blocks(block_class, architecture)
        AvePool_kernel = architecture['AvePool_kernel']
        pool = nn.AvgPool2d(AvePool_kernel)
        flatten = nn.Flatten()
        fc = nn.Linear(in_features=architecture['fc'], out_features=architecture['num_classes'])
        mudules = [cnn] + blocks + [bn, relu, pool, flatten, fc]
        self.modules_list = nn.ModuleList(mudules)
        
    def make_extended_blocks(self, block_class, architecture):
        blocks = []
        for extended_block in architecture['blocks']:
            in_block = extended_block['in']
            out_block = extended_block['out']
            block_size = extended_block['block_size'] 
            for n in range(block_size):
                block = block_class(in_block, out_block, architecture['Pdropout'])
                blocks.append(block)
                in_block = out_block
        return blocks
    
    def init_model_params(self):
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,x):
        for module in self.modules_list:
            x = module(x)
        return x
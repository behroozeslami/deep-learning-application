# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:54:32 2023

@author: eslamimossallamb
"""

import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

class DownConv(nn.Module):
    
    def __init__(self, cin, cout):
        super(DownConv,self).__init__()
        conv1 = nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(cout)
        conv2 = nn.Conv2d(in_channels=cout, out_channels=cout, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(cout)
        self.block = nn.Sequential(*[conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU()])
    
    def forward(self, x):
        
        return self.block(x)
    
class UpConv(nn.Module):
    
    def __init__(self, cin, cout):
        super(UpConv,self).__init__()
        conv1 = nn.Conv2d(in_channels=cin, out_channels=2*cout, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(2*cout)
        conv2 = nn.Conv2d(in_channels=2*cout, out_channels=2*cout, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(2*cout)
        upconv = nn.ConvTranspose2d(in_channels=2*cout, out_channels=cout, kernel_size=2, stride=2)
        self.block = nn.Sequential(*[conv1, bn1, nn.ReLU(), conv2, bn2, nn.ReLU(), upconv])
    
    def forward(self, x):
        
        return self.block(x)
    
class Unet(nn.Module):
    
    def __init__(self, cin=3, features=[32, 64, 128, 256, 512], num_classes=1):
        super(Unet, self).__init__()
        self.downconvs = nn.ModuleList()
        for cout in features:
            self.downconvs.append(DownConv(cin, cout))
            cin = cout
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.connector = UpConv(cin, cin)
        features.reverse()
        self.upconvs = nn.ModuleList()
        cin = 2*cin
        for cout in features[:-1]:
            self.upconvs.append(UpConv(cin, cout//2))
            cin = cout
        cout = features[-1]
        self.downconv_out = DownConv(cin, cout)
        self.conv1 = nn.Conv2d(in_channels=cout, out_channels=num_classes, kernel_size=1, padding=0)
        
            
    def forward(self, x):
        out = x
        downconvs_out = []
        for block in self.downconvs:
            out = block(out)
            downconvs_out.append(out)
            out = self.pool(out)
        downconvs_out.reverse()
        out = self.connector(out)
        
        for index in range(len(downconvs_out)):
            crop = CenterCrop((out.shape[2],out.shape[3]))
            out = torch.concat((crop(downconvs_out[index]), out), 1)
            if index<len(self.upconvs):
                block = self.upconvs[index]
                out = block(out)
        out = self.downconv_out(out)
        crop = CenterCrop((x.shape[2],x.shape[3]))
        out = crop(out)
        out = self.conv1(out)
        
        return out
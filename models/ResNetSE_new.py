#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *
#import librosa
import numpy as np
#import matlab.engine
import pyworld as pw
import scipy
#import py
#eng = matlab.engine.start_matlab()
#eng.cd(r'/home/jiachen/asv/voxceleb_trainer/cqcc_util')


from models.core import parallel


def get_ap(x,index):
    fr = 16000
    f0, sp, ap = pw.wav2world(x, fr,512,10)
    
    return ap


def get_sp(x,index):
    fr = 16000
    f0, sp, ap = pw.wav2world(x, fr,512,10)
    return sp


class ResNetSENEW(nn.Module):
    def __init__(self, block, layers, num_filters, nOut, encoder_type='SAP', **kwargs):

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))
        
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        super(ResNetSENEW, self).__init__()

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=7, stride=(2, 1), padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.maxPool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1), padding=1)
        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        self.avgpool = nn.AvgPool2d((9, 1), stride=1)
        #self.adtavgpool = nn.AdaptiveAvgPool2d((1,None)) 

        self.instancenorm = nn.InstanceNorm1d(257)

        if self.encoder_type == "SAP":
            self.sap_linear = nn.Linear(num_filters[3] * block.expansion, num_filters[3] * block.expansion)
            self.attention = self.new_parameter(num_filters[3] * block.expansion, 1)
            out_dim = num_filters[3] * block.expansion
        else:
            raise ValueError('Undefined encoder')

        self.fc = nn.Linear(out_dim, nOut)
        self.fc2 = nn.Linear(nOut, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

        
    def forward(self, x, feature='stft'):

        #print(x.shape)  #(b,32240)
        #x = self.torchfb(x)+1e-6
        #print(x.shape) #(b,40,202)
        #x = self.instancenorm(x.log()).unsqueeze(1).detach()
        
        
        # for ap and sp
        x = x.cpu().numpy().astype('double')
        
        #inp = []
        #for i in range(len(x)):
        #    inp.append(get_sp(x[i],0))
        
        inp = parallel(get_ap, x)
        x = torch.FloatTensor(inp).unsqueeze(1).detach().cuda().permute(0,1,3,2)
        
        #print(x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        #print(x.shape)

        if self.encoder_type == "SAP":
            x = x.permute(0, 2, 1, 3)
            x = x.squeeze(dim=1).permute(0, 2, 1)  # batch * L * D
            h = torch.tanh(self.sap_linear(x))
            w = torch.matmul(h, self.attention).squeeze(dim=2)
            w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
            x = torch.sum(x * w, dim=1)

        else:
            raise ValueError('Undefined encoder')

            
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        
        return x

def ResNetSE_new(nOut=512, **kwargs):
    # Number of filters
    num_filters = [16, 32, 64, 128]
    model = ResNetSENEW(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    model = nn.DataParallel(model)
    #model = ResNetSE(SEBasicBlock, [2, 2, 2, 2], num_filters, nOut, **kwargs)
    return model

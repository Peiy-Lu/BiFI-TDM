import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _quadruple
from functools import reduce

class MFSEComputation(nn.Module):
    def __init__(self, in_channels=64,out_channels=64, kernel_size=(5, 5), padding=2,
                 planes=[64, 64, 64, 64, 64], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(MFSEComputation, self).__init__()

        d = 32
        self.M = 2
        self.out_channels = out_channels

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.fc1 = nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))
    
        self.fc2 = nn.Conv2d(d,out_channels*self.M,1,1,bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[4]),
                                        nn.ReLU(inplace=True))

    def forward(self, x):
        batch_size = x.size(0)
        
        output = []
        output.append(self.conv11(x))
        output.append(self.conv12(x))

        x = reduce(lambda a, b: a + b,output)
        b, c, h, w = x.shape
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # b, cuv, h, w
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)

        x = x * identity.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v

        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)

        x = self.conv1x1_in(x)  # [80, 640, hw, 25] -> [80, 64, HW, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)

        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        x = self.global_pool(x)
        
        x = self.fc1(x)
        x = self.fc2(x)

        x = x.view(batch_size, self.M, self.out_channels, -1)
        x = self.softmax(x)

        x = list(x.chunk(self.M,dim=1))
        x = list(map(lambda a : a.reshape(batch_size,self.out_channels,1,1) , x))
        V = list(map(lambda a, b : a * b, output, x))
        V = reduce(lambda a, b : a + b, V)

        return V

class MFSE(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.encoder_dim = 64
        self.stride, self.kernel_size, self.padding = (1, 1, 1), (5, 5), 2
        self.planes=[64, 64, 64, 64, 64]
        self.mfseComputation = MFSEComputation(kernel_size=self.kernel_size, padding=self.padding, planes=self.planes, stride=self.stride)
        self.args = args

    def forward(self, input):
        x = self.mfseComputation(input)
        x = x + input
        if self.args.model == 'FRN':
            x = F.relu(x, inplace=True)
        return x


class ConvBlock(nn.Module):
    
    def __init__(self,input_channel,output_channel):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=3,padding=1),
            nn.BatchNorm2d(output_channel))

    def forward(self,inp):
        return self.layers(inp)


class BackBone(nn.Module):

    def __init__(self,num_channel=64, args=None):
        super().__init__()
        
        self.layers = nn.Sequential(
            ConvBlock(3,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ConvBlock(num_channel,num_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        self.args = args
        self.mfse = MFSE(self.args)

    def forward(self,inp):
        return self.mfse(self.layers(inp))
    

if __name__ == '__main__':
    inp = torch.rand(100, 3, 84, 84)
    model = BackBone() 
    output = model(inp)
    print(output.shape)
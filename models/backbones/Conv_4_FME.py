import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FMEComputation(torch.nn.Module):
    def __init__(self, kernel_sizes=[3, 3], planes=[16, 1]):
        super().__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()

        for i in range(num_layers):
            ch_in = 1 if i == 0 else planes[i - 1]
            ch_out = planes[i]
            k_size = kernel_sizes[i]
            nn_modules.append(SepConv4d(in_planes=ch_in, out_planes=ch_out, ksize=k_size, do_padding=True))
            if i != num_layers - 1:
                nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        return x


class SepConv4d(nn.Module):
    def __init__(self, in_planes, out_planes, stride=(1, 1, 1), ksize=3, do_padding=True, bias=False):
        super(SepConv4d, self).__init__()
        self.isproj = False
        padding1 = (0, ksize // 2, ksize // 2) if do_padding else (0, 0, 0)
        padding2 = (ksize // 2, ksize // 2, 0) if do_padding else (0, 0, 0)

        if in_planes != out_planes:
            self.isproj = True
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=bias, padding=0),
                nn.BatchNorm2d(out_planes))

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(1, ksize, ksize),
                      stride=stride, bias=bias, padding=padding1),
            nn.BatchNorm3d(in_planes))
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_planes, out_channels=in_planes, kernel_size=(ksize, ksize, 1),
                      stride=stride, bias=bias, padding=padding2),
            nn.BatchNorm3d(in_planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape
        x = self.conv2(x.view(b, c, u, v, -1))
        b, c, u, v, _ = x.shape
        x = self.relu(x)
        x = self.conv1(x.view(b, c, -1, h, w))
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class FME(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.cca_module = FMEComputation(kernel_sizes=[3, 3], planes=[16, 1])
        self.args = args

    def get_4d_correlation_map(self, spt, qry):
        way = spt.shape[0]
        num_qry = qry.shape[0]

        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)

        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)
        return similarity_map_einsum


    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x

    def forward(self, inp):
        way_ = inp[1]
        shot = inp[2]
        inp = inp[0]

        spt = inp[:way_ * shot]
        qry = inp[way_ * shot:]

        spt = self.normalize_feature(spt)
        qry = self.normalize_feature(qry)

        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)

        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)

        corr4d_s = F.softmax(corr4d_s / 0.2, dim=2)
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / 0.2, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        attn_s = corr4d_s.sum(dim=[4, 5])   # num_qry, way, H_s, W_s, H_q, W_q -> num_qry, way, H_s, W_s
        attn_q = corr4d_q.sum(dim=[2, 3])   # num_qry, way, H_s, W_s, H_q, W_q -> num_qry, way, H_q, W_q

        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)

        spt_attended = spt_attended.view(num_qry, way_ * shot, *spt_attended.shape[2:])
        qry_attended = qry_attended.view(num_qry, way_ * shot, *qry_attended.shape[2:])
        spt_attended = spt_attended.mean(dim=0)
        qry_attended = qry_attended.mean(dim=1)

        if self.args.model == 'FRN':
            spt_attended = F.relu(spt_attended, inplace=True)
            qry_attended = F.relu(qry_attended, inplace=True)

        return torch.cat([spt_attended, qry_attended], dim=0)


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
        self.fme = FME(self.args)

    def forward(self,inp):
        way = inp[1]
        shot = inp[2]
        return self.fme([self.layers(inp[0]),way,shot])
    

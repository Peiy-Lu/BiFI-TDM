import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4, ResNet, Conv_4_FME, Conv_4_MFSE, ResNet_FME, ResNet_MFSE
from .TDM import TDM
import math

def pdist(x,y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist

class BiFI_TDM(nn.Module):
    
    def __init__(self,way=None,shots=None,resnet=False, args=None):
        
        super().__init__()
        
        self.resolution = 25
        self.args = args
        self.args.model = 'Proto'
        
        if resnet:
            self.num_channel = 640
            self.dim = 640
            self.feature_extractor_1 = ResNet_FME.resnet12(args=self.args)
            self.feature_extractor_2 = ResNet_MFSE.resnet12(args=self.args)
        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor_1 = Conv_4_FME.BackBone(self.num_channel, args=self.args)
            self.feature_extractor_2 = Conv_4_MFSE.BackBone(self.num_channel, args=self.args)

        self.shots = shots
        self.way = way
        self.resnet = resnet


        self.TDM = TDM(self.args)
        self.W = nn.Parameter(torch.full((4, 1), 1. / 4), requires_grad=True)
        self.scale = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

        self.M = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.fc1 = nn.Linear(self.num_channel, args.num_class)
        self.fc2 = nn.Linear(self.num_channel, args.num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
            

    def get_feature_vector(self,inp):
        feature_map_1 = self.feature_extractor_1(inp)
        feature_map_2 = self.feature_extractor_2(inp[0])

        return feature_map_1, feature_map_2
    
    def mutual_tdm(self, way, d, support, query):
        query_num = query.shape[1]

        weight = self.TDM(support, query.transpose(0, 1))
        weight = weight.view(way, query_num, d, 1)
        centroid = support.mean(dim=1).unsqueeze(dim=1)
        centroid = centroid * weight
        query = query * weight

        if self.resnet:
            centroid = centroid.mean(dim=-1)
            query = query.mean(dim=-1)
        else:
            centroid = centroid.view(way, -1, self.dim)
            query = query.view(-1, query_num, self.dim)

        l2_dist = torch.sum(torch.pow(centroid - query, 2), dim=-1).transpose(0, 1)
        neg_l2_dist = l2_dist.neg()

        return neg_l2_dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, is_train=False):

        feature_map_1, feature_map_2 = self.get_feature_vector([inp,way,shot])
        _, d, h, w = feature_map_1.shape
        m = h * w
        support_1 = feature_map_1[:way * shot].view(way, shot, d, m)
        query_1 = feature_map_1[way * shot:].view(1, -1, d, m)
        support_2 = feature_map_2[:way * shot].view(way, shot, d, m)
        query_2 = feature_map_2[way * shot:].view(1, -1, d, m)

        neg_l2_dist_1 = self.mutual_tdm(way, d, support_1, query_1)
        neg_l2_dist_2 = self.mutual_tdm(way, d, support_2, query_2)
        neg_l2_dist_3 = self.mutual_tdm(way, d, support_1, query_2)
        neg_l2_dist_4 = self.mutual_tdm(way, d, support_2, query_1)

        neg_l2_dist = self.W[0] * neg_l2_dist_1 + self.W[1] * neg_l2_dist_2 + self.W[2] * neg_l2_dist_3 + self.W[3] * neg_l2_dist_4

        if is_train:
            return neg_l2_dist,\
                    self.fc1(F.relu(feature_map_1).mean(dim=-1).mean(dim=-1)),\
                    self.fc2(F.relu(feature_map_2).mean(dim=-1).mean(dim=-1))
        else:
            return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot
                                           )

        _, max_index = torch.max(neg_l2_dist, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist, fc1, fc2 = self.get_neg_l2_dist(inp=inp,
                                           way=self.way,
                                           shot=self.shots[0],
                                           query_shot=self.shots[1],
                                           is_train=True
                                           )

        logits = neg_l2_dist / self.dim * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction, fc1, fc2, self.M

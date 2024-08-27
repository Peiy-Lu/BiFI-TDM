import torch
import sys
sys.path.append('../../../../')
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones import ResNet, Conv_4
from .TDM import TDM
from numpy import random
import scipy.misc
from PIL import Image


def customLoss(f1, f2, lambda_orth=0.1):
    B, C, H, W = f1.shape

    f1_flat = f1.view(B, C, -1).mean(-1)  # Shape (B, C, H*W)
    f2_flat = f2.view(B, C, -1).mean(-1)  # Shape (B, C, H*W)

    # Orthogonality loss
    cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
    score = cos(f1_flat, f2_flat)
    orth_loss = torch.mean(score)

    total_loss = lambda_orth * orth_loss

    return total_loss


class BiFI_TDM(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False, num_cat=None, args=None):

        super().__init__()

        if resnet:
            num_channel = 640
            self.feature_extractor_1 = ResNet.resnet12()
            self.feature_extractor_2 = ResNet.resnet12()
            self.Woodbury = False
        else:
            num_channel = 64
            self.feature_extractor_1 = Conv_4.BackBone(num_channel)
            self.feature_extractor_2 = Conv_4.BackBone(num_channel)
            self.Woodbury = True


        self.args = args
        self.shots = shots
        self.way = way
        self.resnet = resnet

        self.d = num_channel

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.W = nn.Parameter(torch.full((4, 1), 1. / 4), requires_grad=True)
        self.amplitude = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

        self.resolution = 25

        self.r = nn.Parameter(torch.zeros(8), requires_grad=not is_pretraining)

        if is_pretraining:
            self.num_cat = num_cat
            self.cat_mat_1 = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)
            self.cat_mat_2 = nn.Parameter(torch.randn(self.num_cat, self.resolution, self.d), requires_grad=True)

        self.TDM = TDM(self.args)
        self.M = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.fc1 = nn.Linear(num_channel, args.num_class)
        self.fc2 = nn.Linear(num_channel, args.num_class)


    def get_feature_map(self, inp):
        feature_map_1 = self.feature_extractor_1(inp[0])
        feature_map_1=F.normalize(feature_map_1, dim=1, p=2)
        feature_map_2 = self.feature_extractor_2(inp[0])
        feature_map_2=F.normalize(feature_map_2, dim=1, p=2)
        return feature_map_1, feature_map_2

    def get_recon_dist(self, query, support, alpha, beta, Woodbury):
        way, shot, resolution, d = support.shape
        query_num = query.size(1)

        support = support.view(way, shot * resolution, d)
        query = query.view(-1, query_num * resolution, d)

        reg = support.size(1) / support.size(2)

        lam = reg * alpha.exp() + 1e-6

        rho = beta.exp()

        st = support.permute(0, 2, 1)  # way, d, shot*resolution

        if Woodbury:
            sts = st.matmul(support)  # way, d, d
            m_inv = (sts + torch.eye(sts.size(-1)).to(sts.device).unsqueeze(0).mul(lam)).inverse()  # way, d, d
            hat = m_inv.matmul(sts)  # way, d, d

        else:
            sst = support.matmul(st)  # way, shot*resolution, shot*resolution
            m_inv = (sst + torch.eye(sst.size(-1)).to(sst.device).unsqueeze(0).mul(lam)).inverse() 
            hat = st.matmul(m_inv).matmul(support)  # way, d, d

        Q_bar = query.matmul(hat).mul(rho)  # way, way*query_shot*resolution, d
        
        support = support.view(way, shot, resolution, d)
        query = query.view(-1, query_num, resolution, d)
        weight = self.TDM(support, query)
        weight = weight.view(way, query_num, 1, d)

        Q_bar = Q_bar.reshape(way, query_num, resolution, d)
        query = query.reshape(-1, query_num, resolution, d)
        Q_bar = Q_bar * weight
        query = query * weight
        Q_bar = Q_bar.reshape(way, query_num * resolution, d)
        query = query.reshape(-1, query_num * resolution, d)
        
        dist = (Q_bar - query).pow(2).sum(2).permute(1, 0)
        return dist

    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False, Woodbury=True):
        batch_size = inp.size(0)
        resolution = self.resolution
        d = self.d
        alpha = self.r[:4]
        beta = self.r[4:]

        feature_map_1, feature_map_2 = self.get_feature_map([inp,way,shot])

        customloss = customLoss(feature_map_1, feature_map_2, self.args.cL_scale)

        feature_map_1 = feature_map_1.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()
        self_map = feature_map_2
        self_map = F.normalize(self_map, dim=1, p=2)
        feature_map_2 = self_map.view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()
        support_1 = feature_map_1[:way * shot].view(way, shot * resolution, d)
        support_1_rc = support_1.reshape(way, shot, resolution, d)
        query_1 = feature_map_1[way * shot:].view(1, way * query_shot, resolution, d)
        support_2 = feature_map_2[:way * shot].view(way, shot * resolution, d)
        support_2_rc = support_2.reshape(way, shot, resolution, d)
        query_2 = feature_map_2[way * shot:].view(1, way * query_shot, resolution, d)

        recon_dist_1 = self.get_recon_dist(query=query_1, support=support_1_rc, alpha=alpha[0], beta=beta[0], Woodbury=Woodbury)
        recon_dist_2 = self.get_recon_dist(query=query_2, support=support_2_rc, alpha=alpha[1], beta=beta[1], Woodbury=Woodbury)  
        recon_dist_3 = self.get_recon_dist(query=query_1, support=support_2_rc, alpha=alpha[2], beta=beta[2], Woodbury=Woodbury)  
        recon_dist_4 = self.get_recon_dist(query=query_2, support=support_1_rc, alpha=alpha[3], beta=beta[3], Woodbury=Woodbury)  

        neg_l2_dist_1 = recon_dist_1.neg().view(way * query_shot, resolution, way).mean(1)  # way*query_shot, way
        neg_l2_dist_2 = recon_dist_2.neg().view(way * query_shot, resolution, way).mean(1)  
        neg_l2_dist_3 = recon_dist_3.neg().view(way * query_shot, resolution, way).mean(1)  
        neg_l2_dist_4 = recon_dist_4.neg().view(way * query_shot, resolution, way).mean(1)  
        neg_l2_dist = self.W[0] * neg_l2_dist_1 + self.W[1] * neg_l2_dist_2 + self.W[2] * neg_l2_dist_3 + self.W[3] * neg_l2_dist_4

        if return_support:
            return neg_l2_dist, (support_1,support_2), \
                self.fc1(F.relu(feature_map_1.mean(dim=1))), \
                self.fc2(F.relu(feature_map_2.mean(dim=1))), \
                customloss
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

        neg_l2_dist, s, fc1, fc2, cl = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True,
                                                    Woodbury=self.Woodbury,
                                                    )

        logits = neg_l2_dist * self.scale

        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction, s, fc1, fc2, self.M, cl

import math

import torch
import torch.nn as nn
from typing import Sequence


class MMTMBi(nn.Module):
    """
    2 modal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: dimension of table data
        dim_img: dimension of MIL data
        ratio
        """
        super(MMTMBi, self).__init__()
        dim = dim_tab + dim_img
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )
        self.fc_extract = nn.Sequential(
            nn.Linear(dim_out, dim_img)
        )
        self.fc_tab = nn.Linear(dim_out, dim_tab)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_list) -> Sequence[torch.Tensor]:
        data_list = [x for x in data_list if x is not None]
        img_feat, tab_feat = data_list

        squeeze = torch.cat([tab_feat, img_feat], dim=1)
        excitation = self.fc_squeeze(squeeze)

        tab_out = self.fc_tab(self.relu(excitation))
        tab_out = self.sigmoid(tab_out)

        excitation = self.fc_extract(excitation)
        return excitation, tab_out * tab_feat


class MMTMTri(nn.Module):
    """
    3 modal fusion
    """

    def __init__(self, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: dimension of table feature
        dim_img: dimension of MIL feature
        ratio
        """
        super(MMTMTri, self).__init__()
        dim = dim_img * 3
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )

        self.fc_extract = nn.Sequential(
            nn.Linear(dim_out, dim_img)
        )

    def forward(self, data_list) -> Sequence[torch.Tensor]:
        """

        Args:
            data_list:

        Returns:
            fused feature, tab feature
        """
        data_list = [x for x in data_list if x is not None]
        img_feat_scale1, img_feat_scale2, img_feat_scale3 = data_list
        squeeze = torch.cat([img_feat_scale1, img_feat_scale2, img_feat_scale3], dim=1)
        excitation = self.fc_squeeze(squeeze)
        excitation = self.fc_extract(excitation)
        return excitation, None


class MMTMQuad(nn.Module):
    """
    4 modal fusion
    """

    def __init__(self, dim_tab, dim_img, ratio=4):
        """

        Parameters
        ----------
        dim_tab: dimension of table feature
        dim_img: dimension of MIL feature
        ratio
        """
        super(MMTMQuad, self).__init__()
        dim = dim_tab + dim_img * 3
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Sequential(
            nn.Linear(dim, dim_out),
            nn.LeakyReLU(),
            nn.Linear(dim_out, dim_out)
        )

        self.fc_extract = nn.Sequential(
            nn.Linear(dim_out, dim_img)
        )
        self.fc_tab = nn.Linear(dim_out, dim_tab)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data_list) -> Sequence[torch.Tensor]:
        data_list = [x for x in data_list if x is not None]
        img_feat_scale1, img_feat_scale2, img_feat_scale3, tab_feat = data_list
        squeeze = torch.cat([tab_feat, img_feat_scale1, img_feat_scale2, img_feat_scale3], dim=1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        tab_out = self.fc_tab(excitation)
        tab_out = self.sigmoid(tab_out)

        excitation = self.fc_extract(excitation)

        return excitation, tab_out * tab_feat


class FusionConcat(nn.Module):
    def __init__(self, dim):
        super(FusionConcat, self).__init__()
        self.fc = nn.Linear(dim, dim)

    def forward(self, data_list):
        data_list = [x for x in data_list if x is not None]
        data = torch.cat(data_list)
        return self.fc(data), None


class WSIFusion(nn.Module):
    def __init__(self, dim):
        super(WSIFusion, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, data):
        b, *__ = data.shape
        data_source = data
        data_fc = self.fc(data)
        data_fc = data_fc / math.sqrt(b)
        data_soft_max = torch.softmax(data_fc, dim=0)
        data_agg = torch.sum(data_soft_max * data_source, dim=0, keepdim=True)
        return data_agg


class InstanceAttentionGate(nn.Module):
    def __init__(self, feat_dim):
        super(InstanceAttentionGate, self).__init__()
        self.global_feat_trans = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
        self.trans = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.LeakyReLU(),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, instance_feature, global_feature):
        b, w = instance_feature.shape

        global_feature = self.global_feat_trans(global_feature)

        global_feature = global_feature.repeat(b, 1)
        feat = torch.cat([instance_feature, global_feature], dim=1)
        attention = self.trans(feat)
        attention = attention / math.sqrt(b) # transformer code
        attention = torch.softmax(attention, dim=0)
        ret = torch.sum(instance_feature * attention, dim=0, keepdim=True)
        return ret, attention

import torch
import torch.nn as nn
from rich import print
from models.tabnet.tab_network import TabNet
from models.fusion_refactor import MMTMBi, MMTMTri, MMTMQuad, FusionConcat, InstanceAttentionGate, WSIFusion
from models.ordinalreg import CumulativeLinkLoss, AscensionCallback, LogisticCumulativeLink
from itertools import repeat
from copy import deepcopy
from models.loss import FocalLossV1, LabelSmoothing

class MILFusion(nn.Module):
    def __init__(self, img_feat_input_dim=512, tab_feat_input_dim=32,
                 img_feat_rep_layers=4,
                 num_modal=2,
                 use_tabnet=False,
                 tab_indim=0,
                 local_rank=0,
                 cat_idxs=None,
                 cat_dims=None,
                 num_class=1,
                 lambda_sparse=1e-3,
                 tab_loss_weight=1.0,
                 use_ord=False,
                 use_focal=False,
                 use_smooth=False,
                 fusion='mmtm',
                 ):
        super(MILFusion, self).__init__()
        self.num_modal = num_modal
        self.local_rank = local_rank
        self.use_tabnet = use_tabnet
        self.tab_indim = tab_indim
        self.lambda_sparse = lambda_sparse
        self.num_class = num_class
        self.tab_loss_weight = tab_loss_weight

        if not self.use_tabnet:
            self.tab_loss_weight = 0.
        if use_ord:
            self.crit = CumulativeLinkLoss()
            self.link = LogisticCumulativeLink(num_classes=num_class)
            self.call_back = AscensionCallback()
            fake_num_class = 1
        else:
            self.link = lambda x: x
            self.call_back = None
            fake_num_class = self.num_class
            if self.num_class == 1:
                self.crit = nn.BCEWithLogitsLoss()
            else:
                self.crit = nn.CrossEntropyLoss()

        if use_focal:
            self.crit = FocalLossV1()

        if use_smooth:
            print(f'Label smoothing')
            self.crit = LabelSmoothing()

        self.fusion_method = fusion

        """
        tabular branch
        """
        if self.use_tabnet:
            self.tabnet = TabNet(input_dim=tab_indim, output_dim=fake_num_class,
                                 n_d=32, n_a=32, n_steps=5,
                                 gamma=1.5, n_independent=2, n_shared=2,
                                 momentum=0.3,
                                 cat_idxs=cat_idxs, cat_dims=cat_dims)

        else:
            self.tabnet = None

        if self.use_tabnet and num_modal == 1:
            self.only_tabnet = True
        else:
            self.only_tabnet = False

        """
        only tabnet
        """
        if self.only_tabnet:
            self.feature_fine_tuning = None
        else:
            """pretrained feature fine tune"""
            feature_fine_tuning_layers = []
            for _ in range(img_feat_rep_layers):
                feature_fine_tuning_layers.extend([
                    nn.Linear(img_feat_input_dim, img_feat_input_dim),
                    nn.LeakyReLU(),
                ])

            # 3 three image modals
            self.feature_fine_tuning = nn.ModuleList()
            self.feature_fine_tuning.append(deepcopy(nn.Sequential(*feature_fine_tuning_layers)))
            if self.num_modal == 4 or self.num_modal == 3:
                for _ in range(2):
                    # two image modals
                    self.feature_fine_tuning.append(deepcopy(nn.Sequential(*feature_fine_tuning_layers)))

        if self.only_tabnet or self.num_modal == 3:
            self.table_feature_ft = lambda x: x
        else:
            """tab feature fine tuning"""
            self.table_feature_ft = nn.Sequential(
                nn.Linear(tab_feat_input_dim, tab_feat_input_dim)
            )

        """modal fusion"""
        if self.num_modal == 3:
            # image only
            tab_feat_input_dim = 0
        if not self.only_tabnet:
            if self.fusion_method == 'concat':
                self.fusion_out_dim = tab_feat_input_dim + img_feat_input_dim * (num_modal - 1)
                self.fusion_block = FusionConcat(dim=tab_feat_input_dim + img_feat_input_dim * (num_modal - 1))
            elif self.num_modal == 2 and self.fusion_method == 'mmtm':
                self.fusion_out_dim = (img_feat_input_dim ) * (num_modal - 1) + tab_feat_input_dim
                self.fusion_block = MMTMBi(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
            elif self.num_modal == 3 and self.fusion_method == 'mmtm':
                self.fusion_out_dim = (img_feat_input_dim ) * 3
                self.fusion_block = MMTMTri(dim_img=img_feat_input_dim)
            elif self.num_modal == 4 and self.fusion_method == 'mmtm':
                self.fusion_out_dim = (img_feat_input_dim ) * (num_modal - 1) + tab_feat_input_dim
                self.fusion_block = MMTMQuad(dim_tab=tab_feat_input_dim, dim_img=img_feat_input_dim)
            elif self.num_modal == 1:
                self.fusion_block = None
            else:
                raise NotImplementedError(f'num_modal {num_modal} not implemented')
        else:
            self.fusion_block = lambda x: x

        if self.only_tabnet:
            self.wsi_fusion_blocks = None
        else:
            self.wsi_fusion_blocks = nn.ModuleList()
            self.wsi_fusion_blocks.append(WSIFusion(dim=img_feat_input_dim))
            if self.num_modal == 4 or self.num_modal == 3:
                for _ in range(2):
                    self.wsi_fusion_blocks.append(WSIFusion(dim=img_feat_input_dim))


        if self.only_tabnet:
            self.instance_gate = None
        else:
            self.instance_gate = nn.ModuleList()
            self.instance_gate.append(InstanceAttentionGate(feat_dim=img_feat_input_dim))
            if self.num_modal == 4 or self.num_modal == 3:
                for _ in range(2):
                    self.instance_gate.append(InstanceAttentionGate(feat_dim=img_feat_input_dim))

        """classifier layer"""
        if self.only_tabnet:
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.fusion_out_dim, self.fusion_out_dim),
                nn.Dropout(0.5),
                nn.Linear(self.fusion_out_dim, fake_num_class)
            )

    def forward(self, data):
        if self.use_tabnet:
            tab_data = data['tab_data'].cuda(self.local_rank)
            if self.only_tabnet:
                tab_logit, _, M_loss = self.tabnet(tab_data)
                tab_feat = None
            else:
                tab_logit, tab_feat, M_loss = self.tabnet(tab_data)
        else:
            tab_data = None
            tab_feat = None
            tab_logit = torch.zeros((1, self.num_class)).cuda(self.local_rank)
            M_loss = 0.


        y = data['label'].cuda(self.local_rank)
        mc_y = data['mc_label'].cuda(self.local_rank)
        if self.num_class == 1:
            target = y.view(-1, 1).float()
        elif self.num_class == 2:
            target = y.view(-1).long()
        else:
            target = mc_y.view(-1).long()

        tab_logit = self.link(tab_logit)
        # print(target)
        if self.only_tabnet:
            return tab_logit.detach(), target, self.crit(tab_logit, target), self.call_back, self.link, None

        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(self.local_rank)
        wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(self.local_rank)
        wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(self.local_rank)

        wsi_source_feat = [wsi_feat_scale1, wsi_feat_scale2, wsi_feat_scale3]
        wsi_source_feat = [x.squeeze(0) for x in wsi_source_feat]

        wsi_ft_feat = [feature_tune(x) for feature_tune, x in zip(self.feature_fine_tuning, wsi_source_feat)]
        #
        wsi_feat_global_step1 = [block(x) for block, x in zip(self.wsi_fusion_blocks, wsi_ft_feat)]

        fused_feat, fused_tab_feat = self.fusion_block(wsi_feat_global_step1+[tab_feat])

        wsi_feat_global_step2_and_attention = [instance_fuse(x, global_x)
                                 for instance_fuse, x, global_x in zip(self.instance_gate, wsi_ft_feat, repeat(fused_feat))]

        wsi_feat_global_step2 = [x[0] for x in wsi_feat_global_step2_and_attention]
        attention_weight_all = [x[1] for x in wsi_feat_global_step2_and_attention]

        fused_tab_feat = self.table_feature_ft(fused_tab_feat)

        final_feat = [*wsi_feat_global_step2, fused_tab_feat]
        final_feat = [x for x in final_feat if x is not None]
        final_feat = torch.cat(final_feat, dim=1)
        out = self.classifier(final_feat)
        out = self.link(out)


        loss = self.crit(out, target) + \
               self.tab_loss_weight * self.crit(tab_logit, target) - \
               self.lambda_sparse * M_loss

        return out, target, loss, self.call_back, self.link, attention_weight_all

    def get_params(self, base_lr):
        print(f'Base lr for model: {base_lr}')
        ret = []

        if isinstance(self.tabnet, nn.Module):
            ret.append({
                'params': self.tabnet.parameters(),
                'lr': base_lr
            })
        cls_learning_rate_rate = 100
        if self.classifier is not None:
            classifier_params = []
            for param in self.classifier.parameters():
                classifier_params.append(param)
            ret.append({
                'params': classifier_params,
                'lr': base_lr / cls_learning_rate_rate,
            })

        # misc_params = []
        tab_learning_rate_rate = 100
        if isinstance(self.table_feature_ft, nn.Module):
            misc_params = []
            for param in self.table_feature_ft.parameters():
                misc_params.append(param)
            ret.append({
                'params': misc_params,
                'lr': base_lr / tab_learning_rate_rate,
            })

        mil_learning_rate_rate = 1000
        for part in [self.feature_fine_tuning,
                     self.wsi_fusion_blocks,
                     self.instance_gate,
                     ]:
            if isinstance(part, nn.Module):
                misc_params = []
                for param in part.parameters():
                    misc_params.append(param)
                ret.append({
                    'params': misc_params,
                    'lr': base_lr / mil_learning_rate_rate,
                })

        misc_learning_rate_rate = 100

        for part in [self.fusion_block, ]:
            if isinstance(part, nn.Module):
                misc_params = []
                for param in part.parameters():
                    misc_params.append(param)
                ret.append({
                    'params': misc_params,
                    'lr': base_lr / misc_learning_rate_rate,
                })
        return ret
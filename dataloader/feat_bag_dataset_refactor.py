import os.path as osp
import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from rich import print


class ModalFusionDataset(data_utils.Dataset):
    """"
    Load multi modal data
    """
    def __init__(self,
                 cli_data: pd.DataFrame,
                 scale1_feat_root, scale2_feat_root: None, scale3_feat_root: None,
                 select_scale: int,
                 cfg: Any,
                 shuffle_bag=False, is_train=False):
        """

        Args:
            cli_data:
            scale1_feat_root:
            scale2_feat_root:
            scale3_feat_root:
            select_scale:
            cfg:
            shuffle_bag:
            is_train:
        """
        super(ModalFusionDataset, self).__init__()


        self.pids = cli_data.pid.values.tolist()
        self.targets = cli_data.target.values.tolist()
        try:
            self.mc_targets = cli_data.multi_class_target.values.tolist()
        except:
            print(f'Cant load multi class target')
            self.mc_targets = np.zeros(len(self.targets))

        self.cli_data_df = cli_data
        self.cli_data_cols = [x for x in cli_data.columns if x not in ['pid', 'split', 'target', 'multi_class_target', 'clinical_stage_convert', 'convert_histological_type']]

        if cfg.local_rank == 0:
            pass
        self.scale1_feat_root = scale1_feat_root
        self.scale2_feat_root = scale2_feat_root
        self.scale3_feat_root = scale3_feat_root
        self.select_scale = select_scale

        self.cfg = cfg
        self.shuffle_bag = shuffle_bag
        self.is_train = is_train

        exist_targets = []
        exist_mc_target = []
        exist_pids = []
        miss_cnt = 0
        miss_list = []
        for idx, pid in enumerate(self.pids):
            bag_fp = osp.join(self.scale1_feat_root, f'{pid}.pkl')
            if osp.exists(bag_fp):  # and len(os.listdir(bag_fp)) > 0:
                exist_pids.append(pid)
                exist_targets.append(self.targets[idx])
                exist_mc_target.append(self.mc_targets[idx])
            else:
                miss_cnt += 1
                miss_list.append(pid)

        if cfg.local_rank == 0:
            print(f'{self.cli_data_cols}')
            print(f'Tab input dim => {len(self.cli_data_cols)}')
            print(f'Total : {len(self.pids)}, found {len(exist_pids)}, miss {miss_cnt}')
            print(f'Miss: {miss_list}')

            unique, counts = np.unique(exist_mc_target, return_counts=True)
            value_count = np.asarray((unique, counts)).T
            print(f'{value_count}')
        self.pids = exist_pids
        self.targets = exist_targets
        self.mc_targets = exist_mc_target

    @property
    def tab_data_shape(self):
        return len(self.cli_data_cols)

    @property
    def labels(self):
        return self.mc_targets

    def __len__(self):
        # return 64
        return len(self.pids)

    def load_feat_and_aug(self, bag_fp) -> np.ndarray:
        """
        load WSI feature bag
        Args:
        bag_fp:

        Returns:
        """
        with open(bag_fp, 'rb') as infile:
            bag_feat_list_obj = pickle.load(infile)

        bag_feat = []
        feat_names = []
        for aug_feat_dict in bag_feat_list_obj:
            if self.is_train:
                aug_feat = aug_feat_dict['tr']
                aug_feat = np.vstack([aug_feat, np.expand_dims(aug_feat_dict['val'], 0)])
                random_row = np.random.randint(0, aug_feat.shape[0])
                choice_feat = aug_feat[random_row]
                bag_feat.append(choice_feat)
            else:
                aug_feat = aug_feat_dict['val']
                bag_feat.append(aug_feat)

            feat_names.append(aug_feat_dict['feat_name'])

        del bag_feat_list_obj
        if len(bag_feat) == 0:
            print(f'Empty : {bag_fp}')
            bag_feat = np.zeros((1, 1280))
        else:
            bag_feat = np.vstack(bag_feat)

        if self.is_train:
            if np.random.rand() < 0.5:
                num_of_drop_columns = np.random.randint(0, 100)
                for _ in range(num_of_drop_columns):
                    random_drop_column = np.random.randint(0, bag_feat.shape[1])
                    bag_feat[:, random_drop_column] = 0
            if np.random.rand() < 0.5:
                noise = np.random.normal(loc=0, scale=0.01, size=bag_feat.shape)
                bag_feat += noise

        if self.shuffle_bag:
            instance_size = bag_feat.shape[0]
            shuffle_idx = np.random.permutation(instance_size)
            bag_feat = bag_feat[shuffle_idx]

        return bag_feat, feat_names

    def __getitem__(self, idx) -> Dict:
        try:
            c_pid = self.pids[idx]
        except:
            print('not_found', len(self), idx)
        label = self.targets[idx]
        mc_label = self.mc_targets[idx]
        ret = {}
        if self.select_scale == 0:
            for idx, feat_root in enumerate([self.scale1_feat_root, self.scale2_feat_root, self.scale3_feat_root]):
                bag_fp = osp.join(feat_root, f'{c_pid}.pkl')
                if osp.exists(bag_fp):
                    bag_feat, feat_name = self.load_feat_and_aug(bag_fp)
                else:
                    bag_feat = np.zeros((1, 1280))
                    feat_name = []
                k = f'wsi_feat_scale{idx+1}'
                ret[k] = torch.from_numpy(bag_feat).float()
                ret[k+'_feat_name'] = feat_name
        else:
            if self.select_scale == 1:
                feat_root = self.scale1_feat_root
            elif self.select_scale == 2:
                feat_root = self.scale2_feat_root
            elif self.select_scale == 3:
                feat_root = self.scale3_feat_root
            else:
                feat_root = 'not_exist_path'

            bag_fp = osp.join(feat_root, f'{c_pid}.pkl')
            if osp.exists(bag_fp):
                bag_feat, feat_name = self.load_feat_and_aug(bag_fp)
            else:
                bag_feat = np.zeros((1, 1280))
                feat_name = []
            ret['wsi_feat_scale1'] = torch.from_numpy(bag_feat).float()
            ret['wsi_feat_scale1_feat_name'] = feat_name

            zero_feat = np.zeros((1, 1280))
            ret['wsi_feat_scale2'] = torch.from_numpy(zero_feat).float()
            ret['wsi_feat_scale2_feat_name'] = []
            ret['wsi_feat_scale3'] = torch.from_numpy(zero_feat).float()
            ret['wsi_feat_scale3_feat_name'] = []


        tab_data = self.cli_data_df[self.cli_data_df.pid == c_pid][self.cli_data_cols].values[0]
        ret['name'] = c_pid
        ret['tab_data'] = torch.from_numpy(tab_data).float()
        ret['label'] = torch.tensor(label).float()
        ret['mc_label'] = torch.tensor(mc_label).float()
        ret['pid'] = c_pid
        return ret

def mixup_data(x, alpha=1.0, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.shape[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x

class EMPOBJ:
    def __init__(self):
        self.local_rank = 0

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('./path/to/tabdata.csv')
    bag_feat_root = "/path/to/bag_feature"
    cfg = EMPOBJ()
    from rich.progress import track
    ds = ModalFusionDataset(
        cli_feat=df,
        scale1_feat_root='/path/to/scale1_feature',
        scale2_feat_root='/path/to/scale2_feature',
        scale3_feat_root='/path/to/scale3_feature',
        select_scale=0,
        cfg=cfg,
        shuffle_bag=True,
        is_train=True
    )
    local_rank = 0
    dl = data_utils.DataLoader(ds, num_workers=4)
    for data in track(dl):
        tab_feat = data['tab_feat'].cuda(local_rank)
        wsi_feat_scale1 = data['wsi_feat_scale1'].cuda(local_rank)
        wsi_feat_scale2 = data['wsi_feat_scale2'].cuda(local_rank)
        wsi_feat_scale3 = data['wsi_feat_scale3'].cuda(local_rank)

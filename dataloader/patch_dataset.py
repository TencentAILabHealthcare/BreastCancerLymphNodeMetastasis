import os.path as osp
from glob import glob
import pandas as pd
import torch
import torch.utils.data as data_utils
import numpy as np
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as alb


class PatchDataset(data_utils.Dataset):

    def __init__(self, patch_root_dir):
        patch_dirs = glob(osp.join(patch_root_dir, '*'))
        patch_dirs = [x for x in patch_dirs if osp.isdir(x)]

        img_fps = []
        for dir in patch_dirs:
            img_fp = glob(osp.join(dir, '*'))
            img_fp = [x for x in img_fp if not osp.isdir(x)]
            img_fps.extend(img_fp)

        self.img_fps = img_fps

        self.trans = alb.Compose([
            alb.Resize(512, 512),
            alb.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.img_fps)
        pass

    def __getitem__(self, item):
        fp = self.img_fps[item]
        try:
            img = cv2.imread(fp)[:, :, ::-1]
        except:
            print(f'cant read {fp}')
            img = np.zeros((512, 512, 3))

        img = self.trans(image=img)['image']
        pid = osp.basename(osp.dirname(fp))
        img_fp = osp.basename(fp)
        return {
            'img': img,
            'pid': pid,
            'img_fp': img_fp
        }
        pass

val_trans = alb.Compose([
    alb.Resize(512, 512),
    alb.Normalize(),
    ToTensorV2(),
])


class MILPatchDataset(data_utils.Dataset):
    def __init__(self, patch_root_dir, df: pd.DataFrame, shuffle_bag=False, is_train=False, max_patch_num=180):

        self.patch_root_dir = patch_root_dir
        self.pids = df['pid'].values.tolist()
        self.targets = df['target'].values.tolist()

        self.shuffle_bag = shuffle_bag

        exist_pids = []
        exist_targets = []
        miss_cnt = 0
        for idx, pid in enumerate(self.pids):
            bag_fp = osp.join(self.patch_root_dir, pid)
            if osp.exists(bag_fp):
                img_files = glob(osp.join(bag_fp,'*.png'))
                if len(img_files) < 2:
                    miss_cnt += 1
                else:
                    exist_pids.append(pid)
                    exist_targets.append(self.targets[idx])
            else:
                miss_cnt += 1

        self.trans = val_trans

        print(f'Total : {len(self.pids)}, found {len(exist_pids)}, miss {miss_cnt}')

        self.pids = exist_pids
        self.max_patch_num = max_patch_num
        self.targets = exist_targets

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid_dir = osp.join(self.patch_root_dir, self.pids[idx])
        img_fp_list = glob(osp.join(pid_dir, '*.png'))

        if self.shuffle_bag:
            np.random.shuffle(img_fp_list)

        img_list = []
        for fp in img_fp_list:
            try:
                img = cv2.imread(fp)[:,:,::-1]
            except:
                print(f'Cant read {fp}')
                img = (np.ones((512, 512, 3)) * 255).astype('uint8')
            img = self.trans(image=img)['image']
            img_list.append(img)

        img_tensor = torch.stack(img_list)

        pid = osp.basename(pid_dir)
        return {
            'data': img_tensor.float(),
            'name': pid,
            'label': self.targets[idx],
            'pid': pid
        }

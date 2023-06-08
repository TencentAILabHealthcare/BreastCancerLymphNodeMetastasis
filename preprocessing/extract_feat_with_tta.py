import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import os.path as osp
import torch
import pandas as pd
from glob import glob
import torch.utils.data as data_utils
from tqdm import tqdm
import pickle
import numpy as np
from multiprocessing.pool import ThreadPool, Pool
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as alb
import random
import queue
import threading
from models.effnet import EffNet
from concurrent.futures import ThreadPoolExecutor
import argparse

val_trans = alb.Compose([
    alb.Resize(512, 512),
    alb.Normalize(),
    ToTensorV2(),
])

TTA_TIMES = 1


class MILPatchDataset(data_utils.Dataset):
    def __init__(self, img_fp_list):
        self.img_fp_list = img_fp_list

    def __len__(self):
        return len(self.img_fp_list)

    def __getitem__(self, idx):
        img_fp = self.img_fp_list[idx]
        img = cv2.imread(img_fp)[:, :, ::-1]
        pid = osp.basename(osp.dirname(img_fp))
        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]

        val_img = val_trans(image=img)['image']

        return pid, img_bname, val_img



gt_csv_fp = "/path/to/dataset_with_wsi.csv"


def merge_feat_to_bag(feat_dir):
    """
    all patch feature were saved in their pid dir,
    this function extract feature from pid dir and merge them into a single bag file
    Args:
        feat_dir: instance level feature save path
    Returns:
    """
    dataset_df = pd.read_csv(gt_csv_fp)
    pid_label_mp = dict(zip(dataset_df['pid'], dataset_df['target']))
    pid_dirs = glob(osp.join(feat_dir, '*'))
    pid_dirs = [d for d in pid_dirs if osp.isdir(d)]
    for pid_d in pid_dirs:
        feats = []
        f_names = []
        feat_fps = glob((osp.join(pid_d, '*.pkl')))
        for feat_fp in feat_fps:
            with open(feat_fp, 'rb') as infile:
                try:
                    feat_data = pickle.load(infile)
                except:
                    continue
            feats.append(feat_data)
            f_names.append(osp.basename(feat_fp).rsplit('.', 1)[0])

        feats = np.stack(feats)
        pid = osp.basename(pid_d)
        save_fp = osp.join(feat_dir, f'{pid}.pkl')
        print(f'Save features bag to: {save_fp} with size: {feats.shape}')

        if pid not in pid_label_mp.keys():
            continue
        with open(save_fp, 'wb') as outfile:
            pickle.dump({
                'feat_bag': feats,
                'feat_name': f_names,
                'bag_label': pid_label_mp[pid]
            },
                outfile)

def save_feat_in_thread(batch_pid, batch_img_bname, batch_val_feat):
    for b_idx, (pid, img_bname, val_feat) in enumerate(zip(batch_pid, batch_img_bname, batch_val_feat)):
        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)
        feat_save_name = osp.join(feat_save_dir, f'{img_bname}.pkl')
        save_dict = {}
        save_dict['val'] = val_feat
        with open(feat_save_name, 'wb') as outfile:
            pickle.dump(save_dict, outfile)

def pred_and_save_with_dataloader(model, img_fp_list, local_rank):
    random.seed(42)
    model.cuda(local_rank)
    model.eval()
    executor = ThreadPoolExecutor(max_workers=16)
    dl = torch.utils.data.DataLoader(
        MILPatchDataset(img_fp_list),
        batch_size=1,
        num_workers=0,
        shuffle=False
    )
    for batch in dl:

        batch_pid, batch_img_bname, batch_val_img = batch

        batch_val_img = batch_val_img.cuda(local_rank)

        with torch.no_grad():
            batch_val_feat = model(batch_val_img)

            executor.submit(save_feat_in_thread, batch_pid, batch_img_bname, batch_val_feat)

img_queue = queue.Queue(maxsize=128)
img_fp_queue = queue.Queue()


def read_worker():
    while True:
        imgfp = img_fp_queue.get()
        if imgfp is None:
            break
        try:
            img = cv2.imread(imgfp)[:, :, ::-1]
        except:
            img = np.zeros((512, 512,3), dtype='uint8')
        aug_img_list = []

        img_queue.put((imgfp, img, aug_img_list))

def save_pkl_file(feat_save_name, save_dict):
    if osp.exists(feat_save_name):
        try:
            with open(feat_save_name, 'rb') as infile:
                old_save_dict = pickle.load(infile)
        except:
            old_save_dict = {}
    else:
        old_save_dict = {}

    with open(feat_save_name, 'wb') as outfile:
        pickle.dump(save_dict, outfile)

def pred_and_save(model, img_fp_list, local_rank):

    total = len(img_fp_list)
    read_cnt = 0
    for img_fp in img_fp_list:
        img_fp_queue.put(img_fp)

    threads = []
    num_worker_threads = 36
    for _ in range(num_worker_threads):
        t = threading.Thread(target=read_worker)
        t.start()
        threads.append(t)


    random.seed(42)
    model.cuda(local_rank)
    model.eval()

    executor = ThreadPoolExecutor(max_workers=16)

    while read_cnt < total:

        img_fp, img, aug_img_list = img_queue.get()
        read_cnt += 1
        pid = osp.basename(osp.dirname(img_fp))

        img_bname = osp.basename(img_fp).rsplit('.', 1)[0]

        feat_save_dir = osp.join(save_dir, pid)
        os.makedirs(feat_save_dir, exist_ok=True)

        feat_save_name = osp.join(feat_save_dir, f'{img_bname}.pkl')

        save_dict = {}
        with torch.no_grad():

            val_img = val_trans(image=img)['image'].cuda(local_rank)

            val_feat = model(val_img.unsqueeze(0)).detach().cpu().numpy()

            save_dict['val'] = val_feat

        executor.submit(save_pkl_file, feat_save_name, save_dict)

    # stop workers
    for i in range(num_worker_threads):
        img_fp_queue.put(None)

    for t in threads:
        t.join()


def main():
    """
    extract patch feature from WSI
    save each patch feature into pid dir
    then merge them into a single file
    """
    print(f'Load dataset...')

    model = EffNet()


    img_fp_list = []
    print(f'Working on {patch_root_dir}')
    bag_fp_list = glob(osp.join(patch_root_dir, '*'))
    for bag_fp in bag_fp_list:
        img_files = glob(osp.join(bag_fp, '*.png'))
        img_fp_list.extend(img_files)

    print(f'Len of img {len(img_fp_list)}')

    local_rank = list(range(8))

    img_fp_list = sorted(img_fp_list)

    np.random.shuffle(img_fp_list)
    num_processes = 8
    num_train_images = len(img_fp_list)
    images_per_process = num_train_images / num_processes

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        tasks.append((model, img_fp_list[start_index:end_index], (num_process - 1) % 4))
        if start_index == end_index:
            print("Task #" + str(num_process) +
                  ": Process slide " + str(start_index))
        else:
            print("Task #" + str(num_process) + ": Process slides " +
                  str(start_index) + " to " + str(end_index))
    # start tasks
    pbar = tqdm(total=len(tasks))

    def update(*a):
        pbar.update()

    results = []
    with Pool(num_processes) as p:
        for _ in tqdm(p.starmap(pred_and_save, tasks), total=len(tasks)):
            pass

if __name__ == '__main__':

    load_and_save_path = {
        'x5': ['/path/to/5x/patches',
               '/path/to/5x/features'],
        'x10': ['/path/to/10x/patches',
                '/path/to/10x/features'],
        'x20': ['/path/to/20x/patches',
                '/path/to/20x/features']
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='x5')
    arg = parser.parse_args()

    select = load_and_save_path[arg.level]

    patch_root_dir, save_dir = select
    main()

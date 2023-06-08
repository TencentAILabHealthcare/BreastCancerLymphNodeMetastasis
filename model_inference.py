# System libs
import argparse
import datetime
import os
import os.path as osp
import pickle
from typing import Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as data_utils
from rich import print
from sklearn import metrics

from configs.defaults import _C as train_config
from dataloader.feat_bag_dataset_refactor import ModalFusionDataset
from metrics.roc_auc_refactor import ROC_AUC
from models.mil_net_refactor import MILFusion
from utils import AverageMeter, setup_logger
from utils_lib.epoch_gather import EpochGather

matplotlib.use("Agg")

def print_in_main_thread(msg: str,):
    if local_rank == 0:
        print(msg)

def log_in_main_thread(msg: str):
    if local_rank == 0:
        logger.info(msg)

def evaluate(model: nn.Module, val_loader, epoch, local_rank, save_header: str, final_test=True, dump_dir=None) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    model
    val_loader
    epoch
    local_rank

    Returns
    -------

    """
    ave_total_loss = AverageMeter()
    auc_meter = ROC_AUC()
    pid_epo_gather = EpochGather()
    model.eval()
    from scipy.sparse import csc_matrix
    from models.tabnet.utils import create_explain_matrix
    res_explain = []
    reducing_matrix = create_explain_matrix(
        model.module.tabnet.input_dim,
        model.module.tabnet.cat_emb_dim,
        model.module.tabnet.cat_idxs,
        model.module.tabnet.post_embed_dim,
    )

    feature_importances_ = np.zeros((model.module.tabnet.post_embed_dim))

    with torch.no_grad():
        for batch_nb, batch_data in enumerate(val_loader):
            # print(batch_data['pid'])
            pid_epo_gather.update(batch_data['pid'])

            logit, label, loss, *__ = model(batch_data)

            M_explain, masks = model.module.tabnet.forward_masks(batch_data['tab_data'].cuda())
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), reducing_matrix
                )

            res_explain.append(
                csc_matrix.dot(M_explain.cpu().detach().numpy(), reducing_matrix)
            )

            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

            output_prob = torch.softmax(logit.detach(), dim=1)
            batch_size = logit.shape[0]
            auc_meter.update([output_prob.cpu(), label.view(-1).cpu()])
            ave_total_loss.update(loss.data.item())

            del label
            del batch_data

        feature_importances_ = csc_matrix.dot(
            feature_importances_, reducing_matrix
        )

        res_explain = np.vstack(res_explain)

    if dump_dir is not None:
        d = osp.join(dump_dir, f'{save_header}_feat_importance_{local_rank}.npy')
        np.save(d, feature_importances_)
    """
    generate feature importance
    """

    val_explain_matrix, val_masks = res_explain, res_masks

    num_of_class = output_prob.shape[1]

    pid_in_cgpu = np.array(pid_epo_gather.val)
    pred_in_cgpu = torch.cat(auc_meter.predictions).cpu().numpy()
    target_in_cgpu = torch.cat(auc_meter.targets).cpu().numpy()

    current_fold_val_data = np.concatenate([np.array(pid_in_cgpu).reshape(-1, 1),
                           pred_in_cgpu.reshape(-1, num_of_class),
                           target_in_cgpu.reshape(-1, 1)],
                          -1)

    pred_label_names = [f'Class_{i}' for i in range(num_of_class)]
    col_names = ['pid', *pred_label_names, 'target']

    current_fold_val_data = np.hstack([current_fold_val_data, val_explain_matrix])

    num_steps = len(val_masks)
    num_mat_cols = val_explain_matrix.shape[1]

    col_names += [f'explain_matrix_col_{i}' for i in range(num_mat_cols)]
    for step_idx in range(num_steps):
        col_names += [f'explain_matrix_at_step_{step_idx}_col_{i}' for i in range(num_mat_cols)]
        current_fold_val_data = np.hstack([current_fold_val_data, val_masks[step_idx]])


    df = pd.DataFrame(current_fold_val_data, columns=col_names)

    if dump_dir is not None:
        dump_fp = osp.join(dump_dir, f'{save_header}_pred_{local_rank}.csv')

        df.to_csv(dump_fp, index=False, encoding='utf_8_sig')
        logger.info(f'Dump tabnet feature result to : {dump_fp}')


    # summary
    val_loss = ave_total_loss.average()
    val_auc, all_pred, all_target = auc_meter.compute()

    logger.info(f'ROC AUC: {val_auc:.5f}')

    all_pred_label = np.argmax(all_pred, axis=1)
    all_target_label = np.argmax(all_target, axis=1)

    num_class = all_pred.shape[1]

    if final_test == True and local_rank == 0:
        logger.info(f"Num of class: {num_class} ")
        if num_class == :
            f1_scores = metrics.f1_score(all_target_label, all_pred_label, average='macro')
            precision_scores = metrics.precision_score(all_target_label, all_pred_label, average='macro')
            recall_scores = metrics.recall_score(all_target_label, all_pred_label, average='macro')

            logger.info(f"Best precision: {precision_scores} ")
            logger.info(f"Best recall: {recall_scores} ")
            logger.info(f'Best F1-Score: {f1_scores}')

            logger.info(f'Cal AUC of Different Class')
            num_of_class = all_pred.shape[1]
            for cls_idx in range(num_of_class):
                current_prob = all_pred[:, cls_idx]
                current_label = (all_target_label == cls_idx)
                current_auc = metrics.roc_auc_score(current_label, current_prob)
                logger.info(f'OVR AUC of Class {cls_idx} {current_auc:.5f}')

    if final_test:
        logger.info(f'Gathering pid from all subprocess')
        all_pid = pid_epo_gather.gather()
        num_of_class = all_pred.shape[1]

    if final_test and local_rank == 0:
        data = np.concatenate([np.array(all_pid).reshape(-1, 1),
                         all_pred.reshape(-1, num_of_class),
                         all_target_label.reshape(-1, 1)],
                        -1)

        pred_label_names = [f'Class_{i}' for i in range(num_of_class)]
        col_names = ['pid', *pred_label_names, 'target']
        df = pd.DataFrame(data, columns=col_names)
        dump_fp = osp.join(dump_dir, f'{save_header}_pred.csv')

        df.to_csv(dump_fp, index=False, encoding='utf_8_sig')
        logger.info(f'Dump label pred result to : {dump_fp}')

    else:
        pass

    all_target_label = np.argmax(all_target, axis=1)
    val_acc = (all_pred_label == all_target_label).astype('float').mean()
    if local_rank == 0:
        progress = {"step": epoch, "type": "test", "loss": val_loss,
                    "acc": val_acc, 'auc': val_auc}
        logger.info(
            f'\n{metrics.classification_report(all_target_label, all_pred_label)} \n'
            f'{metrics.confusion_matrix(all_target_label, all_pred_label)}')

        logger.info(f'Epoch: {epoch}, Val Loss: {val_loss:.6f}, AUC: {val_auc:.6f}, ACC:{val_acc:.6f}')
    return val_loss, val_auc, val_acc

def main(cfg, local_rank):
    """
    build
    prepare model training
    :param cfg:
    :param local_rank:
    :return:
    """
    if local_rank == 0:
        logger.info(f'Build model')

    with open(cfg.dataset.tab_data_path, 'rb') as infile:
        tab_data = pickle.load(infile)
    cat_dims = tab_data['cat_dims']
    cat_idxs = tab_data['cat_idxs']

    if local_rank == 0:
        print(f'Tabnet config dims & idx: {cat_dims}  {cat_idxs}')

    tab_data_df = pd.read_csv(cfg.dataset.tab_data_path.rsplit('.', 1)[0] + '.csv')

    test_data_df = tab_data_df[tab_data_df.split == 'test']

    if local_rank == 0:
        logger.info(f'Build dataset')

    test_dataset = ModalFusionDataset(
        cli_data=test_data_df,
        scale1_feat_root=cfg.dataset.scale1_feat_root,
        scale2_feat_root=cfg.dataset.scale2_feat_root,
        scale3_feat_root=cfg.dataset.scale3_feat_root,
        select_scale=cfg.dataset.select_scale,
        cfg=cfg,
        shuffle_bag=False,
        is_train=False
    )

    # add data sampler
    log_in_main_thread('Dataset load finish')

    test_sampler = data_utils.distributed.DistributedSampler(test_dataset, rank=local_rank)

    num_workers = cfg.test.workers

    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=num_workers,
        drop_last=False,
        shuffle=False,
        pin_memory=False,
        sampler=test_sampler
    )

    log_in_main_thread('Build model')
    if hasattr(cfg.model, 'fusion_method'):
        fusion = cfg.model.fusion_method
    else:
        fusion = 'mmtm'
    if hasattr(cfg.model, 'use_k_agg'):
        use_k_agg = cfg.model.use_k_agg
        k_agg = cfg.model.k_agg
    else:
        use_k_agg = False
        k_agg = 10

    if hasattr(cfg.train, 'use_focal'):
        use_focal = cfg.train.use_focal
    else:
        use_focal = False

    if hasattr(cfg.train, 'use_smooth'):
        use_smooth = cfg.train.use_smooth
    else:
        use_smooth = False

    model = MILFusion(img_feat_input_dim=1280,
                      tab_feat_input_dim=32,
                      img_feat_rep_layers=4,
                      num_modal=cfg.model.num_modal,
                      fusion=fusion,
                      num_class=cfg.model.num_class,
                      use_ord=cfg.model.use_ord,
                      use_focal=use_focal,
                      use_smooth=use_smooth,
                      use_tabnet=cfg.model.use_tabnet,
                      tab_indim=test_dataset.tab_data_shape,
                      cat_dims=cat_dims,
                      cat_idxs=cat_idxs,
                      local_rank=local_rank)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(local_rank)
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    model_path = cfg.checkpoint
    """
    load best ckpt
    """

    logger.info(f'Load best model from {model_path}')

    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    model.load_state_dict(torch.load(model_path, map_location=map_location))

    test_loss, test_auc, test_acc = evaluate(model, test_loader, cfg.train.num_epoch, local_rank,
                                             save_header='Test', final_test=True,
                                             dump_dir=cfg.save_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Multi-modal multi-instance model"
    )
    parser.add_argument(
        "--cfg",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    args = parser.parse_args()

    cfg = train_config
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    local_rank = args.local_rank

    torch.multiprocessing.set_sharing_strategy('file_system')
    # set dist
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank)

    print(f'local rank: {args.local_rank}')
    time_now = datetime.datetime.now()

    cfg.save_dir = osp.join(cfg.save_dir,
                            f'{time_now.year}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}')

    if not os.path.isdir(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)
    logger = setup_logger(distributed_rank=args.local_rank, filename=osp.join(cfg.save_dir, 'train_log.txt'))  # TODO
    log_in_main_thread(f'Save result to : {cfg.save_dir}')

    if args.local_rank == 0:
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("Running with config:\n{}".format(cfg))
        with open(os.path.join(cfg.save_dir, 'config.yaml'), 'w') as f:
            f.write("{}".format(cfg))

    dist.barrier()
    num_gpus = 1

    cfg.local_rank = args.local_rank


    main(cfg, args.local_rank)
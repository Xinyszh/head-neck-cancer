from data.util import get_dataset
from module.util import get_multimodal_model, get_optimizer
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score, average_precision_score, roc_curve
import os
import shutil
from tqdm import tqdm
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='T1-COR-18', help="name of this experiment")
    parser.add_argument('--seed', type=int, default=1, help="1, 2, 3, or any other seeds")
    parser.add_argument('--fuse_tag', type=str, default='Concate', help="Choose a fusion strategy.")
    parser.add_argument('--before_model_tag', type=str, default='ResNet18', help="Choose a model.")
    parser.add_argument('--after_model_tag', type=str, default='ResNet18', help="Choose a model.")
    # parser.add_argument('--DCE_model_tag', type=str, default='ResNet34', help="Choose a model.")
    parser.add_argument('--optimizer_tag', type=str, default='Adam', help="Choose a optimizer.")
    parser.add_argument('--modal', type=str, default='T1_COR', help="Choose modal for experiment.")
    parser.add_argument('--bs', type=int, default=5, help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--lowest_lr', type=float, default=1e-6, help="Learning rate.")
    parser.add_argument('--wd', type=float, default=5e-4, help="Weight decay.")
    parser.add_argument('--num_epochs', type=int, default=150, help="Number of epochs.")
    parser.add_argument('--device', type=int, default=5, help="Choose gpu device.")
    parser.add_argument('--device_gpus', type=list, default=[5,6], help="Choose gpu device.")
    parser.add_argument('--log_dir', type=str, default='/jhcnas1/xinyi/head_neck_cancer/resample', help="Address to store the log files.")
    parser.add_argument('--debug', action='store_true', help='False for saving result or Ture for not saving result.')
    args = parser.parse_args()

    # ------------------------------------ seed, device, log ------------------------------------- #
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    # device = torch.device('cuda')
    exp_name = args.exp_name
    log_dir = args.log_dir

    if not args.debug:
        print('saving the result.')

        writer = SummaryWriter(os.path.join(log_dir, "summary", exp_name))
        os.makedirs(os.path.join(log_dir, "bk", exp_name), exist_ok=True)
        shutil.copyfile('./scripts/train_multi.sh', os.path.join(log_dir, "bk", exp_name, './train_multi.sh'))

    print('Device: {}'.format(args.device))
    print('Experiment: {}'.format(exp_name))
    print('Seed: {}'.format(seed))

    # ------------------------------------ dataloader ------------------------------------- #
    train_dataset = get_dataset(
        args.modal,
        dataset_split="train",
        transform_split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=3,
        pin_memory=False,
        # persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )

    valid_dataset = get_dataset(
        args.modal,
        dataset_split="val",
        transform_split="val"
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        # persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )

    test_dataset = get_dataset(
        args.modal,
        dataset_split="test",
        transform_split="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        # persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )

    # ------------------------------------ model and optimizer ------------------------------------- #
    num_classes = 2
    if args.modal == 'T1_COR':
        before_data_size = (1, 280, 280, 150)
        after_data_size = (1, 280, 280, 150)
    elif args.modal == 'T1_TRA':
        before_data_size = (1, 280, 280, 250)
        after_data_size = (1, 280, 280, 250)
    elif args.modal == 'T2_TRA':
        before_data_size = (1, 280, 280, 250)
        after_data_size = (1, 280, 280, 250)


    model = get_multimodal_model(args.fuse_tag, (args.before_model_tag, args.after_model_tag),
                                 (before_data_size,after_data_size), num_classes)
    # model = nn.DataParallel(model,device_ids=args.device_gpus)
    model.to(device)
    optimizer = get_optimizer(args.optimizer_tag, model, args.lr, args.wd)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lowest_lr)

    result_path = os.path.join(log_dir, "result", exp_name)
    model_path = os.path.join(result_path, "model.th")

    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch']

    def eval(model, data_loader):
        model.eval()
        gts = torch.LongTensor().to(device)
        probs = torch.FloatTensor().to(device)
        preds = torch.FloatTensor().to(device)
        indices = torch.LongTensor().to(device)
        for index, before_data, after_data, label in tqdm(data_loader, leave=False):
            index = index.to(device)
            before_data = before_data.float()
            after_data = after_data.float()
            before_data = before_data.to(device)
            after_data = after_data.to(device)
            # sub = sub.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(before_data, after_data)
                prob = torch.softmax(logit, dim=1)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)

            gts = torch.cat((gts, label), 0)
            probs = torch.cat((probs, prob[:, 1]), 0)  # prob[1] for malignancy
            preds = torch.cat((preds, pred), 0)
            indices = torch.cat((indices, index), 0)

        gts_numpy = gts.cpu().detach().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        preds_numpy = preds.cpu().detach().numpy()
        indices_numpy = indices.cpu().detach().numpy()
        tp = np.sum((preds_numpy == 1) & (gts_numpy == 1))
        fp = np.sum((preds_numpy == 1) & (gts_numpy == 0))
        tn = np.sum((preds_numpy == 0) & (gts_numpy == 0))
        fn = np.sum((preds_numpy == 0) & (gts_numpy == 1))

        accs = (tp + tn) * 1. / len(gts_numpy)
        tpr = tp * 1. / (gts_numpy == 1).sum()  # recall, sensitivity
        tnr = tn * 1. / (gts_numpy == 0).sum()  # specificity
        ppv = tp * 1. / (tp + fp)  # precision
        npv = tn * 1. / (tn + fn)
        f1 = tp * 2. / (tp * 2. + fp + fn)

        aps = average_precision_score(gts_numpy, probs_numpy)
        aucs = roc_auc_score(gts_numpy, probs_numpy)
        model.train()
        return accs, aps, aucs, tp, fp, tn, fn, tpr, tnr, ppv, npv, f1, probs_numpy, preds_numpy, gts_numpy, indices_numpy

    # ----------------------------------- loss function -----------------------------------  #
    # P = (train_dataset.malignant==1).sum()
    # N = (train_dataset.malignant==0).sum()
    # weight = torch.stack([P/(P+N), N/(P+N)])*num_classes
    # weight = weight.to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=weight,reduction='none')
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # ----------------------------------- start training ------------------------------------------- #
    if not args.debug:
        result_path = os.path.join(log_dir, "result", exp_name)
        os.makedirs(result_path, exist_ok=True)
        model_path = os.path.join(result_path, "model.th")
        val_csv_path = os.path.join(result_path, "val_result.csv")
        test_csv_path = os.path.join(result_path, "test_result.csv")
    best_auc = 0

    for epoch in range(start_epoch,args.num_epochs):
        for iter_num, (_, before_data, after_data, label) in tqdm(enumerate(train_loader)):
            step = epoch * len(train_loader) + iter_num
            before_data = before_data.float()
            after_data = after_data.float()
            before_data = before_data.to(device)
            after_data = after_data.to(device)

            label = label.to(device)

            logit = model(before_data, after_data)

            loss = criterion(logit, label).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            main_log_freq = 10
            if step % main_log_freq == 0:
                if not args.debug:
                    writer.add_scalar('loss', loss.detach().cpu(), step)
        writer.add_scalar('epoch_loss', loss.detach().cpu(), epoch)

        if not args.debug:
            # ------------------------------------- validation ----------------------------------------#
            val_acc, val_ap, val_auc, val_tp, val_fp, val_tn, val_fn, val_tpr, val_tnr, val_ppv, val_npv, val_f1, val_probs, val_preds, val_gts, val_idxs = eval(
                model, valid_loader)
            writer.add_scalar("val/1_auc", val_auc, epoch)
            writer.add_scalar("val/2_acc", val_acc, epoch)
            writer.add_scalar("val/3_ap", val_ap, epoch)
            writer.add_scalar("val/4_f1", val_f1, epoch)
            writer.add_scalar("val/5_tpr", val_tpr, epoch)
            writer.add_scalar("val/6_tnr", val_tnr, epoch)
            writer.add_scalar("val/7_ppv", val_ppv, epoch)
            writer.add_scalar("val/8_npv", val_npv, epoch)
            writer.add_scalar("val/9_tp", val_tp, epoch)
            writer.add_scalar("val/10_fp", val_fp, epoch)
            writer.add_scalar("val/11_tn", val_tn, epoch)
            writer.add_scalar("val/12_fn", val_fn, epoch)

            # ----------------------------- save model and results, do test ------------------------------#
            if (val_auc >= best_auc):
                best_auc = val_auc

                # ------------------------------------- test ----------------------------------------#
                te_acc, te_ap, te_auc, te_tp, te_fp, te_tn, te_fn, te_tpr, te_tnr, te_ppv, te_npv, te_f1, te_probs, te_preds, te_gts, te_idxs = eval(
                    model, test_loader)
                writer.add_scalar("test/1_auc", te_auc, epoch)
                writer.add_scalar("test/2_acc", te_acc, epoch)
                writer.add_scalar("test/3_ap", te_ap, epoch)
                writer.add_scalar("test/4_f1", te_f1, epoch)
                writer.add_scalar("test/5_tpr", te_tpr, epoch)
                writer.add_scalar("test/6_tnr", te_tnr, epoch)
                writer.add_scalar("test/7_ppv", te_ppv, epoch)
                writer.add_scalar("test/8_npv", te_npv, epoch)
                writer.add_scalar("test/9_tp", te_tp, epoch)
                writer.add_scalar("test/10_fp", te_fp, epoch)
                writer.add_scalar("test/11_tn", te_tn, epoch)
                writer.add_scalar("test/12_fn", te_fn, epoch)

                state_dict = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler':scheduler.state_dict(),
                }
                with open(model_path, "wb") as f:
                    torch.save(state_dict, f)

                df_val = valid_dataset.df
                df_val['Probability'] = pd.Series(val_probs, index=val_idxs)
                df_val['Prediction'] = pd.Series(val_preds, index=val_idxs)
                df_val['GT'] = pd.Series(val_gts, index=val_idxs)
                df_val.to_csv(val_csv_path, index=False, encoding='utf-8_sig')

                df_test = test_dataset.df
                df_test['Probability'] = pd.Series(te_probs, index=te_idxs)
                df_test['Prediction'] = pd.Series(te_preds, index=te_idxs)
                df_test['GT'] = pd.Series(te_gts, index=te_idxs)
                df_test.to_csv(test_csv_path, index=False, encoding='utf-8_sig')

        scheduler.step()


if __name__ == '__main__':
    train()
# 加载模型



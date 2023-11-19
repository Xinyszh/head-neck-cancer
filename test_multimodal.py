from data.util import get_dataset
from module.util import get_model, get_multimodal_model
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, roc_curve
import os
import shutil
from tqdm import tqdm
import pandas as pd

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
    parser.add_argument('--before_model_tag', type=str, default='ResNet18', help="Choose a model.")
    parser.add_argument('--after_model_tag', type=str, default='ResNet18', help="Choose a model.")
    parser.add_argument('--fuse_tag', type=str, default='Concate', help="Choose a fusion strategy.")
    parser.add_argument('--modal', type=str, default='T1_COR_zhongliu', help="Choose modal for experiment.")
    parser.add_argument('--bs', type=int, default=6, help="Batch size.")
    parser.add_argument('--split', type=str, default='test', help="Choose the dataset to test.")
    parser.add_argument('--save_csv', action='store_true', help='True for saving a csv file.')
    parser.add_argument('--device', type=int, default=0, help="Choose gpu device.")
    parser.add_argument('--log_dir', type=str, default='/jhcnas1/xinyi/head_neck_cancer/test', help="Address to store the log files.")
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

    exp_name = args.exp_name
    log_dir = args.log_dir
    
    print('Device: {}'.format(args.device))
    print('Experiment: {}'.format(exp_name))
    print('Seed: {}'.format(seed))

    # ------------------------------------ dataloader ------------------------------------- #
    
    te_dataset = get_dataset(
        args.modal,
        transform_split="test"
    )

    te_loader = DataLoader(
        te_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )

    # ------------------------------------ model and optimizer ------------------------------------- #

    num_classes = 2
    if args.modal == 'T1_COR_zhongliu':
        before_data_size = (1, 280, 280, 150)
        after_data_size = (1, 280, 280, 150)
    elif args.modal == 'T1_TRA_zhongliu':
        before_data_size = (1, 280, 280, 250)
        after_data_size = (1, 280, 280, 250)
    elif args.modal == 'T2_TRA_zhongliu':
        before_data_size = (1, 280, 280, 250)
        after_data_size = (1, 280, 280, 250)
    model = get_multimodal_model(args.fuse_tag, (args.before_model_tag, args.after_model_tag),
                                 (before_data_size, after_data_size), num_classes)
    # model = nn.DataParallel(model,device_ids=args.device_gpus)
    model.to(device)
    
    result_path = os.path.join(log_dir, "result", exp_name)
    model_path = os.path.join(result_path, "model.th")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])
    model.eval()
    
    # -------------------------------------- evalaution --------------------------------------- #
    def eval(model, data_loader):
        model.eval()
        gts = torch.LongTensor().to(device)
        probs = torch.FloatTensor().to(device)
        preds = torch.FloatTensor().to(device)
        indices = torch.LongTensor().to(device)
        for index, t2, dwi, sub, label in tqdm(data_loader, leave=False):
            index = index.to(device)
            t2 = t2.to(device)
            dwi = dwi.to(device)
            sub = sub.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(t2, dwi, sub)
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
        
        accs = (tp+tn)*1./len(gts_numpy)
        tpr = tp*1. / (gts_numpy == 1).sum()   # recall, sensitivity
        tnr = tn*1. / (gts_numpy == 0).sum()   # specificity
        ppv = tp*1. / (tp+fp)  #precision
        npv = tn*1. / (tn+fn)
        f1 = tp*2./(tp*2.+fp+fn)

        aps = average_precision_score(gts_numpy, probs_numpy)
        aucs = roc_auc_score(gts_numpy, probs_numpy)
        model.train()
        return accs, aps, aucs, tp, fp, tn, fn, tpr, tnr, ppv, npv, f1, probs_numpy, preds_numpy, gts_numpy, indices_numpy

    # ------------------------------------- test ----------------------------------------#
    te_acc, te_ap, te_auc, te_tp, te_fp, te_tn, te_fn, te_tpr, te_tnr, te_ppv, te_npv, te_f1, te_probs, te_preds, te_gts, te_idxs = eval(model, te_loader)
    print('Accuracy:', te_acc)
    print('Average Precision:', te_ap)
    print('AUC:', te_auc)
    print('True Positive Rate:', te_tpr)
    print('True Negative Rate:', te_tnr)
    print('Positive Predictive Value:', te_ppv)
    print('Negative Predictive Value:', te_npv)
    print('F1 Score:', te_f1)
    print('True Positives:', te_tp)
    print('False Positives:', te_fp)
    print('True Negatives:', te_tn)
    print('False Negatives:', te_fn)
    #----------------------------- save model and results, do test ------------------------------#
    #te_csv_path = os.path.join(result_path, "{}_result.csv".format(args.split))
    te_csv_path = os.path.join(result_path, "{}_result.csv".format('test_2'))
    df_te = te_dataset.df
    df_te['Probability'] = pd.Series(te_probs, index=te_idxs)
    df_te['Prediction'] = pd.Series(te_preds, index=te_idxs)
    df_te['GT'] = pd.Series(te_gts, index=te_idxs)
    if args.save_csv:
        df_te.to_csv(te_csv_path, index=False, encoding='utf-8_sig')


if __name__ == '__main__':
    train()
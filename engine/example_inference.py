# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


from utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold

def do_eval(val_loader, model, device, log):
    """
        Eval the FAS model on the target data.
        The function evaluates the model on the target data and return TPR@FPR, HTER
        and AUC.
        Used in train.py to eval the model after each epoch; used in test.py to eval
        the final model.
    """

    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()

    prob_list = []
    label_list = []

    model.eval()
    with torch.no_grad():
        for idx, (img, _, _, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device)

            logits, _, _, _, _ = model(img, img, img, label)#, 'test')

            valid_loss = criterion(logits, label)
            valid_losses.update(valid_loss.item())

            acc = accuracy(logits, label, topk = (1, ))
            valid_top1.update(acc[0].item())

            prob = F.softmax(logits, dim = 1).cpu().data.numpy()[:, 1]
            label = label.cpu().data.numpy()

            prob_list = np.append(prob_list, prob)
            label_list = np.append(label_list, label)


    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    fpr, tpr, thr = roc_curve(label_list, prob_list)
    tpr_filtered = tpr[fpr <= 1 / 100]
    if len(tpr_filtered) == 0:
        rate = 0
    else:
        rate = tpr_filtered[-1]
    
    if log is not None:
        log.write(f"TPR@FPR = {rate}\n", is_file = True)
    else:
        print("TPR@FPR = ", rate)

    return [
        valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid,
        auc_score, threshold, ACC_threshold * 100, rate]

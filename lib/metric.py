
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
#import ipdb

def accuracy(output, target):

    with torch.no_grad():

        num_item = target.size(0)

        pred_mask = output > 0.5
        pred_ = pred_mask.cpu().numpy()
        target_ = target.cpu().numpy()
        acc = (pred_ == target_)
        sum_val = acc.sum()


        return float(sum_val), num_item

def precision(output, target):

    with torch.no_grad():

        pred_mask = output > 0.5

        indx = (pred_mask == 1).nonzero(as_tuple = True)
        target_ = target[indx]

        sum_val = target_.cpu().numpy().sum()
        num_item = target_.size(0)

        return float(sum_val), num_item

def recall(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        pred_mask = output > 0.5

        indx = (target == 1).nonzero(as_tuple = True)
        pred_ = pred_mask[indx]

        sum_val = pred_.cpu().numpy().sum()
        num_item = pred_.size(0)

        return float(sum_val), num_item

def roc(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():

        scores = output[:,0]
        labels = target[:,0]
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc
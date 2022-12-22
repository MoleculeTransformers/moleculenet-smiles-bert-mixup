import numpy as np
from sklearn.metrics import roc_auc_score
from torch import sigmoid
from torch.nn.functional import softmax
import torch


def flat_accuracy(preds, labels):
    """
    Function to calculate the accuracy of our predictions vs labels
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_auroc_score(preds, labels):
    """
    Function to calculate the roc_auc_score of our predictions vs labels
    """

    pred_flat = softmax(torch.tensor(preds), dim=1)[:, 1]
    # labels_flat = np.argmax(labels, axis=1)
    return roc_auc_score(labels, pred_flat.detach().cpu().numpy())

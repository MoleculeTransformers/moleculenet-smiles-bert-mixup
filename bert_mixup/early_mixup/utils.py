import torch
from sklearn.metrics import roc_auc_score
from torch import sigmoid
from torch.nn.functional import softmax


def flat_auroc_score(preds, labels):
    """
    Function to calculate the roc_auc_score of our predictions vs labels
    """
    pred_flat = softmax(preds, dim=1)[:, 1]
    # labels_flat = np.argmax(labels, axis=1)
    return roc_auc_score(labels, pred_flat.detach().cpu().numpy())


def mixup_augment(embedding1, embedding2, label1, label2, lam):
    embedding_output = lam * embedding1 + (1.0 - lam) * embedding2
    label_output = lam * label1 + (1.0 - lam) * label2
    return (embedding_output, label_output)


def get_perm(x, args):
    """get random permutation"""
    batch_size = x.size()[0]
    if args.cuda and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    return index


def mixup_criterion_cross_entropy(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..util import one_hot


def nll_loss(output, target):
    return F.nll_loss(output, target)


def l1loss(output, target):
    return F.l1_loss(output, target)


def smooth_l1loss(output, target, reduction='mean'):
    return F.smooth_l1_loss(output, target, reduction=reduction)


def CEloss(output, target, **kwargs):
    return nn.CrossEntropyLoss(**kwargs)(output, target)


def focal_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = 'none',
        eps: float = 1e-8) -> torch.Tensor:

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}"
                         .format(input.shape))

    if input.size(0) != target.size(0):
        raise ValueError('Expected input batch_size ({}) to match target batch_size ({}).'
                         .format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1],
        device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1., gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}"
                                  .format(reduction))
    return loss


def giou(pred, gt, eps=1e-7):
    # pred, gt in xyhw
    gt_area = gt[2] * gt[3]
    pred_area = pred[2] * pred[3]
    # intersection
    tp_x = torch.maximum(gt[0]-gt[3]/2, pred[0]-pred[3]/2)
    tp_y = torch.maximum(gt[1]-gt[2]/2, pred[1]-pred[2]/2)
    br_x = torch.minimum(gt[0]+gt[3]/2, pred[0]+pred[3]/2)
    br_y = torch.minimum(gt[1]+gt[2]/2, pred[1]+pred[2]/2)
    inter = (br_x - tp_x) * (br_y - tp_y)
    union = gt_area + pred_area - inter
    iou = inter / (union + eps)
    # enclosure
    tp_x = torch.minimum(gt[0]-gt[3]/2, pred[0]-pred[3]/2)
    tp_y = torch.minimum(gt[1]-gt[2]/2, pred[1]-pred[2]/2)
    br_x = torch.maximum(gt[0]+gt[3]/2, pred[0]+pred[3]/2)
    br_y = torch.maximum(gt[1]+gt[2]/2, pred[1]+pred[2]/2)
    enclosure = (br_x - tp_x) * (br_y - tp_y)
    # giou
    giou = iou - (enclosure-union)/(enclosure + eps)

    return giou
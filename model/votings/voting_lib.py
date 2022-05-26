import torch
import numpy as np

from ..util import deparametric, deparametric_ts


def get_center_hm(ys, xs, vals, h, w):
    hm = np.zeros((h, w))
    np.add.at(hm, (ys, xs), vals)
    return hm


def vote_centor(ys, xs, vals, h, w):
    hm = get_center_hm(ys, xs, vals, h, w)
    max_idx = hm.argmax()
    y_pred = max_idx // w
    x_pred = max_idx % w
    return y_pred, x_pred


def vote_hw(arr1d, val1d, maxval):
    hm = np.zeros(maxval)
    np.add.at(hm, arr1d, val1d)
    max_idx = hm.argmax()
    return max_idx


def get_center_predictions(r_pred, cos_pred, sin_pred):
    h, w = r_pred.shape
    widxs, hidxs = np.meshgrid(np.arange(0, w), np.arange(0, h))
    hidxs_pred = hidxs - r_pred*sin_pred
    widxs_pred = widxs - r_pred*cos_pred
    return hidxs_pred, widxs_pred


def get_predictions(output, priors, eps=1e-7):
    """
    output shape:   7xHxW
    """
    objprob_01 = output[5:7] - np.max(output[5:7], axis=0, keepdims=True)
    objprob_pred = np.exp(objprob_01[1]) / np.sum(np.exp(objprob_01), axis=0)
    h_a, w_a, r_a = priors['h_a'], priors['w_a'], priors['r_a']
    r_pred = deparametric(output[0], r_a)
    h_pred = deparametric(output[3], h_a)
    w_pred = deparametric(output[4], w_a)
    cos_pred = output[1]
    sin_pred = output[2]
    theta_unit_pred = np.sqrt(cos_pred**2+sin_pred**2) + eps
    cos_pred = cos_pred / theta_unit_pred
    sin_pred = sin_pred / theta_unit_pred
    return r_pred, cos_pred, sin_pred, h_pred, w_pred, objprob_pred


def get_center_predictions_ts(r_pred, cos_pred, sin_pred):
    h, w = r_pred.shape
    hidxs, widxs = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    hidxs = hidxs.to(r_pred.device)
    widxs = widxs.to(r_pred.device)
    hidxs_pred = hidxs - r_pred*sin_pred
    widxs_pred = widxs - r_pred*cos_pred
    return hidxs_pred, widxs_pred


def get_predictions_ts(output, h_a, w_a, r_a, eps=1e-7):
    """
    output shape:   7xHxW
    """
    objprob_01 = output[5:7] - torch.max(output[5:7], axis=0, keepdims=True)[0]
    objprob_pred = torch.exp(objprob_01[1]) / torch.sum(torch.exp(objprob_01), axis=0)
    r_pred = deparametric_ts(output[0], r_a)
    h_pred = deparametric_ts(output[3], h_a)
    w_pred = deparametric_ts(output[4], w_a)
    cos_pred = output[1]
    sin_pred = output[2]
    theta_unit_pred = torch.sqrt(cos_pred**2+sin_pred**2) + eps
    cos_pred = cos_pred / theta_unit_pred
    sin_pred = sin_pred / theta_unit_pred
    return r_pred, cos_pred, sin_pred, h_pred, w_pred, objprob_pred
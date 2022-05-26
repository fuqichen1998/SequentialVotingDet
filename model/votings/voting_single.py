import torch
import numpy as np

from ..util import xyxy2mask, rchw2xyxy
from .voting_lib import get_predictions, get_center_predictions, vote_centor, vote_hw


def get_obj_predictions(output, handmask, hand_bboxs, priors, eps=1e-7):
    """
    Inputs:
        output shape:       7xHxW (torch tensor or numpy array)
        handmask shape:     HxW (torch tensor or numpy array)
        hand_bboxs:         nhandsx4 (torch tensor or numpy array)
        priors:             {'h_a', 'w_a', 'r_a'} (dictionary)

    Outputs:
        dets:               nhandsx5 (x1, y1, x2, y2, score)
    """
    # to numpy
    if type(output) == torch.Tensor:
        output = output.detach().cpu().numpy()
    if type(handmask) == torch.Tensor:
        handmask = handmask.detach().cpu().numpy()
    if type(hand_bboxs) == torch.Tensor:
        hand_bboxs = hand_bboxs.detach().cpu().numpy()
    
    h, w = handmask.shape

    # get predictions
    r_pred, cos_pred, sin_pred, h_pred, w_pred, objprob_pred = get_predictions(output, priors, eps)
    ys_pred, xs_pred = get_center_predictions(r_pred, cos_pred, sin_pred)

    # for each hands
    nhands = hand_bboxs.shape[0]
    dets = np.zeros((nhands, 5))  # x1, y1, x2, y2, confidence
    for i in range(nhands):
        x1, y1, x2, y2 = hand_bboxs[i]
        single_handmask = xyxy2mask(x1, y1, x2, y2, h, w)
        single_handmaks_nonoverlap = (single_handmask*handmask) == 1
        npixels_hand_nonoverlap = single_handmaks_nonoverlap.sum()

        # if there is no hand pixels for voting
        if npixels_hand_nonoverlap == 0:
            continue

        # only consider pixels inside this bounding box
        this_contact_pred = objprob_pred[single_handmaks_nonoverlap]
        this_ys_pred = np.rint(ys_pred[single_handmaks_nonoverlap]).astype(np.int)
        this_xs_pred = np.rint(xs_pred[single_handmaks_nonoverlap]).astype(np.int)
        this_h_pred = np.rint(h_pred[single_handmaks_nonoverlap]).astype(np.int)
        this_w_pred = np.rint(w_pred[single_handmaks_nonoverlap]).astype(np.int)

        dets[i, 4] = np.mean(this_contact_pred)

        # remove points outside the image
        inside_mask = (0 <= this_ys_pred) * (this_ys_pred < h) * \
            (0 <= this_xs_pred) * (this_xs_pred < w) * \
            (0 <= this_h_pred) * (this_h_pred < h) * \
            (0 <= this_w_pred) * (this_w_pred < w)
        this_contact_pred = this_contact_pred[inside_mask]
        this_ys_pred = this_ys_pred[inside_mask]
        this_xs_pred = this_xs_pred[inside_mask]
        this_h_pred = this_h_pred[inside_mask]
        this_w_pred = this_w_pred[inside_mask]
        
        # if no pixel has a valid vote
        if not this_xs_pred.size:
            continue
        
        dets[i, 4] = np.mean(this_contact_pred)
        b_c_y_pred, b_c_x_pred = vote_centor(this_ys_pred, this_xs_pred, this_contact_pred, h, w)
        b_h_pred = vote_hw(this_h_pred, this_contact_pred, h)
        b_w_pred = vote_hw(this_w_pred, this_contact_pred, w)
        dets[i, :4] = [b_c_y_pred, b_c_x_pred, b_h_pred, b_w_pred]
        
    dets[:, :4] = rchw2xyxy(dets[:, :4], h, w)
    return dets
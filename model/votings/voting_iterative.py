import torch
import numpy as np

from .voting_single import get_obj_predictions
from .voting_lib import get_center_predictions, vote_centor, vote_hw
from ..util import xyxy2mask, rchw2xyxy, xyxy2xyhw, deparametric


def get_obj_predictions_iterative(output, handmask, hand_bboxs, priors, max_vote=5, tol=0.05, eps=1e-7):
    """
    Inputs:
        output shape:       14xHxW (torch tensor or numpy array)
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
    # get the rough obj det (nhands, 5)
    rough_obj_dets = get_obj_predictions(output[:7], handmask, hand_bboxs, priors, eps)

    # get predictions
    h_a, w_a, obj_r_a = priors['h_a'], priors['w_a'], priors['obj_r_a']
    r_pred = deparametric(output[7], obj_r_a)
    h_pred = deparametric(output[10], h_a)
    w_pred = deparametric(output[11], w_a)
    cos_pred = output[8]
    sin_pred = output[9]
    theta_unit_pred = np.sqrt(cos_pred**2+sin_pred**2) + eps
    cos_pred = cos_pred / theta_unit_pred
    sin_pred = sin_pred / theta_unit_pred
    ys_pred, xs_pred = get_center_predictions(r_pred, cos_pred, sin_pred)
    objprob_01 = output[12:14] - np.max(output[12:14], axis=0, keepdims=True) 
    objprob_pred = np.exp(objprob_01[1]) / np.sum(np.exp(objprob_01), axis=0)

    # for each hands
    nhands = hand_bboxs.shape[0]
    dets = np.zeros((nhands, 6))  # x1, y1, x2, y2, contact score, obj prob score
    for i in range(nhands):
        rough_obj_box = rough_obj_dets[i, :4].copy()
        contact_score = rough_obj_dets[i, 4].copy()
        
        # keep the contacting score
        dets[i, 4] = contact_score
        
        # voting
        if rough_obj_box.any():
            vote_count = 1
            while vote_count < max_vote:
                x1, y1, x2, y2 = rough_obj_box
                xc, yc, box_h, box_w = xyxy2xyhw(rough_obj_box)
                single_obj_mask = xyxy2mask(x1, y1, x2, y2, h, w) == 1.
                this_ys_pred = np.rint(ys_pred[single_obj_mask]).astype(np.int)
                this_xs_pred = np.rint(xs_pred[single_obj_mask]).astype(np.int)
                this_h_pred = np.rint(h_pred[single_obj_mask]).astype(np.int)
                this_w_pred = np.rint(w_pred[single_obj_mask]).astype(np.int)
                this_objprob_pred = objprob_pred[single_obj_mask]
                inside_mask = (0 <= this_ys_pred) * (this_ys_pred < h) * \
                    (0 <= this_xs_pred) * (this_xs_pred < w) * \
                    (0 <= this_h_pred) * (this_h_pred < h) * \
                    (0 <= this_w_pred) * (this_w_pred < w)
                this_ys_pred = this_ys_pred[inside_mask]
                this_xs_pred = this_xs_pred[inside_mask]
                this_h_pred = this_h_pred[inside_mask]
                this_w_pred = this_w_pred[inside_mask]
                this_contact_pred = this_objprob_pred[inside_mask]

                # if no pixel has a valid vote
                if not this_xs_pred.size:
                    break
                
                b_c_y_pred, b_c_x_pred = vote_centor(this_ys_pred, this_xs_pred, this_contact_pred, h, w)
                b_h_pred = vote_hw(this_h_pred, this_contact_pred, h)
                b_w_pred = vote_hw(this_w_pred, this_contact_pred, w)
                dets[i, :4] = [b_c_y_pred, b_c_x_pred, b_h_pred, b_w_pred]
                dets[i:i+1, :4] = rchw2xyxy(dets[i:i+1, :4], h, w)
                
                # recompute the score
                single_obj_mask_final = xyxy2mask(*dets[i, :4], h, w) == 1.
                dets[i, 5] = np.mean(objprob_pred[single_obj_mask_final])
                
                # terminate
                if abs(xc-b_c_x_pred)/box_w < tol and abs(yc-b_c_y_pred)/box_h < tol and abs(b_h_pred-box_h)/box_h < tol and abs(b_w_pred-box_w)/box_w < tol:
                    break
                
                vote_count += 1
                rough_obj_box = dets[i, :4].copy()

    return rough_obj_dets, dets
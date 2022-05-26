import torch
from .loss_lib import giou
from ..util import xyxy2mask_ts
from .loss_double_vote import relational_boxfields_loss
from ..votings.voting_general import box_field_voting, maksed_inliers_gaussian_sum
from ..votings.voting_lib import get_predictions_ts, get_center_predictions_ts

def rl_loss(output, target, eps=1e-7):
    
    batch_size, _, h, w = output.shape
    
    device = output.device
    nhands = target['nhands']
    h_a, w_a, r_a, obj_r_a = target['h_a'][0], target['w_a'][0], target['r_a'][0], target['obj_r_a'][0]
    
    enlarged_hand_bboxes = target['padded_enlarged_unqiue_hand_bboxes'].to(device)
    padded_obj_bboxes = target['padded_enlarged_obj_bboxes'].to(device)
    
    total_loss = torch.tensor(0.).to(device)
    total_num_objs = 0
    
    # for each sample
    for b in range(batch_size):
        # if there is no contacts
        if not padded_obj_bboxes[b].any():
            continue
        
        # get predictions
        ## h2o
        h2o_r, h2o_cos, h2osin, h2o_h, h2o_w, contact_prob = get_predictions_ts(output[b, :7], h_a, w_a, r_a, eps)
        h2o_ys, h2o_xs = get_center_predictions_ts(h2o_r, h2o_cos, h2osin)
        ## o2o
        o2o_r, o2o_cos, o2osin, o2o_h, o2o_w, obj_prob = get_predictions_ts(output[b, 7:], h_a, w_a, obj_r_a, eps)
        o2o_ys, o2o_xs = get_center_predictions_ts(o2o_r, o2o_cos, o2osin)
        
        # for each hand
        for i in range(nhands[b]):
            # skip small objects
            if padded_obj_bboxes[b, i, 2] * padded_obj_bboxes[b, i, 3] < 10:
                continue
            
            ox_gt, oy_gt, oh_gt, ow_gt = padded_obj_bboxes[b, i]
            # average of hand pixels
            this_hand_mask = xyxy2mask_ts(*enlarged_hand_bboxes[b, i], h, w).to(device) == 1
            
            # voting
            ox_major, oy_major, oh_major, ow_major = box_field_voting(
                h2o_xs[this_hand_mask],
                h2o_ys[this_hand_mask],
                h2o_h[this_hand_mask],
                h2o_w[this_hand_mask],
                contact_prob[this_hand_mask],
                h, w
            )
            ox_pred, oy_pred, oh_pred, ow_pred, _ = maksed_inliers_gaussian_sum(
                h2o_xs[this_hand_mask],
                h2o_ys[this_hand_mask],
                h2o_h[this_hand_mask],
                h2o_w[this_hand_mask],
                contact_prob[this_hand_mask],
                ox_major, oy_major, oh_major, ow_major
            )
            total_loss += (1 - giou([ox_pred, oy_pred, oh_pred, ow_pred], [ox_gt, oy_gt, oh_gt, ow_gt]))
            
            for _ in range(5): # horizon of 5
                ox1 = ox_pred - ow_pred/2
                oy1 = oy_pred - oh_pred/2
                ox2 = ox_pred + ow_pred/2
                oy2 = oy_pred + oh_pred/2
                this_obj_mask = xyxy2mask_ts(ox1, oy1, ox2, oy2, h, w).to(device) == 1
                ox_major, oy_major, oh_major, ow_major = box_field_voting(
                    o2o_xs[this_obj_mask],
                    o2o_ys[this_obj_mask],
                    o2o_h[this_obj_mask],
                    o2o_w[this_obj_mask],
                    obj_prob[this_obj_mask],
                    h, w
                )
                ox_pred, oy_pred, oh_pred, ow_pred, _ = maksed_inliers_gaussian_sum(
                    o2o_xs[this_obj_mask],
                    o2o_ys[this_obj_mask],
                    o2o_h[this_obj_mask],
                    o2o_w[this_obj_mask],
                    obj_prob[this_obj_mask],
                    ox_major, oy_major, oh_major, ow_major
                )
                total_loss += (1 - giou([ox_pred, oy_pred, oh_pred, ow_pred], [ox_gt, oy_gt, oh_gt, ow_gt]))
            total_num_objs += 1
    if total_num_objs > 0:
        total_loss = total_loss / total_num_objs / 6
    return {'total_loss': total_loss}


def combined_loss(output, target, eps=1e-7):
    fields_losses = relational_boxfields_loss(output, target)
    rl_losses = rl_loss(output, target)
    fields_losses['giou_loss'] = fields_losses['total_loss']
    fields_losses['total_loss'] = fields_losses['total_loss'] + \
        rl_losses['total_loss']
    return fields_losses
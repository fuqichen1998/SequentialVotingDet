import torch
from ..util import xyxy2mask_ts
from .loss_lib import focal_loss, smooth_l1loss


def relational_boxfields_loss(output, target, eps=1e-7):
    """
        output shape:   Nx14xHxW
        target shape:   Nx12xHxW
        handmask shape: NxHxW
        objmask shape:  NxHxW
        
        target channel info: tr, a(h), b(w), th, tw, contact  
    """

    batch_size, _, h, w = output.shape
    
    device = output.device
    GT = target['GT'].to(device)
    handmask = target['handmask'].to(device)
    contact_mask = target['contact_mask'].to(device)
    objbox_mask = target['objbox_mask'].to(device)
    
    nhands = target['nhands']
    n_unique_objs = target['n_unique_objs']
    enlarged_hand_bboxes = target['padded_enlarged_unqiue_hand_bboxes'].to(device)
    enlarged_obj_bboxes = target['padded_enlarged_unqiue_obj_bboxes'].to(device)
    
    total_nonoverlap_hands = 0
    total_nonoverlap_contact_hands = 0
    total_nonoverlap_objs = 0
    
    # hands to objs losses
    tr_loss = torch.tensor(0.).to(device)
    cos_loss = torch.tensor(0.).to(device)
    sin_loss = torch.tensor(0.).to(device)
    th_loss = torch.tensor(0.).to(device)
    tw_loss = torch.tensor(0.).to(device)
    contact_loss = torch.tensor(0.).to(device)
    # objs losses
    obj_tr_loss = torch.tensor(0.).to(device)
    obj_cos_loss = torch.tensor(0.).to(device)
    obj_sin_loss = torch.tensor(0.).to(device)
    obj_th_loss = torch.tensor(0.).to(device)
    obj_tw_loss = torch.tensor(0.).to(device)
    obj_prob_loss = torch.tensor(0.).to(device)
    
    handmask_bn = handmask == 1
    
    # hands to objs predictions
    tr_pred = output[:, 0]
    cos_pred = output[:, 1]
    sin_pred = output[:, 2]
    theta_unit_pred = torch.sqrt(cos_pred**2+sin_pred**2) + eps
    cos_pred = cos_pred / theta_unit_pred
    sin_pred = sin_pred / theta_unit_pred
    th_pred = output[:, 3]
    tw_pred = output[:, 4]
    
    # precompute contact_prob loss
    if handmask_bn.any():
        contact_pred = output[:, 5:7]
        contact_loss_all = focal_loss(contact_pred, GT[:, 5].type(torch.long), 0.25, reduction='none')
    
    # objs predictions
    obj_tr_pred = output[:, 7]
    obj_cos_pred = output[:, 8]
    obj_sin_pred = output[:, 9]
    obj_theta_unit_pred = torch.sqrt(obj_cos_pred**2+obj_sin_pred**2) + eps
    obj_cos_pred = obj_cos_pred / obj_theta_unit_pred
    obj_sin_pred = obj_sin_pred / obj_theta_unit_pred
    obj_th_pred = output[:, 10]
    obj_tw_pred = output[:, 11]
    obj_pred = output[:, 12:14]
    
    # precompute is_obj_loss
    obj_prob_loss = focal_loss(obj_pred, GT[:, 11].type(torch.long), 0.25, reduction='mean')
    
    # for each sample
    for b in range(batch_size):
        this_handmask = handmask[b]
        this_contact_mask = contact_mask[b]
        this_objbox_mask = objbox_mask[b]
        
        this_enlarged_hand_bboxes = enlarged_hand_bboxes[b, :nhands[b]]
        this_enlarged_obj_bboxes = enlarged_obj_bboxes[b, :n_unique_objs[b]]
        
        # for each hand
        for i in range(nhands[b]):
            this_enlarged_hand_mask = xyxy2mask_ts(*this_enlarged_hand_bboxes[i], h, w).to(device)
            this_nonoverlap_hand_mask = (this_handmask * this_enlarged_hand_mask) == 1
            this_hand_contact_mask = (this_contact_mask * this_enlarged_hand_mask) == 1
            # r, h, w
            if this_hand_contact_mask.any():
                this_tr_pred = tr_pred[b][this_hand_contact_mask]
                this_cos_pred = cos_pred[b][this_hand_contact_mask]
                this_sin_pred = sin_pred[b][this_hand_contact_mask]
                this_th_pred = th_pred[b][this_hand_contact_mask]
                this_tw_pred = tw_pred[b][this_hand_contact_mask]

                tr_loss += smooth_l1loss(this_tr_pred, GT[b, 0][this_hand_contact_mask], reduction='mean')
                cos_loss += smooth_l1loss(this_cos_pred, GT[b, 1][this_hand_contact_mask], reduction='mean')
                sin_loss += smooth_l1loss(this_sin_pred, GT[b, 2][this_hand_contact_mask], reduction='mean')
                th_loss += smooth_l1loss(this_th_pred, GT[b, 3][this_hand_contact_mask], reduction='mean')
                tw_loss += smooth_l1loss(this_tw_pred, GT[b, 4][this_hand_contact_mask], reduction='mean')
                
                total_nonoverlap_contact_hands += 1
                
            # contact_prob
            if this_nonoverlap_hand_mask.any():
                contact_loss += contact_loss_all[b][this_nonoverlap_hand_mask].mean()
                total_nonoverlap_hands += 1
        
        # for each object
        for i in range(n_unique_objs[b]):
            this_enlarged_obj_mask = xyxy2mask_ts(*this_enlarged_obj_bboxes[i], h, w).to(device)
            this_nonoverlap_obj_mask = (this_objbox_mask * this_enlarged_obj_mask) == 1
            # r, h, w
            if this_nonoverlap_obj_mask.any():
                this_tr_pred = obj_tr_pred[b][this_nonoverlap_obj_mask]
                this_cos_pred = obj_cos_pred[b][this_nonoverlap_obj_mask]
                this_sin_pred = obj_sin_pred[b][this_nonoverlap_obj_mask]
                this_th_pred = obj_th_pred[b][this_nonoverlap_obj_mask]
                this_tw_pred = obj_tw_pred[b][this_nonoverlap_obj_mask]
                
                obj_tr_loss += smooth_l1loss(this_tr_pred, GT[b, 6][this_nonoverlap_obj_mask], reduction='mean')
                obj_cos_loss += smooth_l1loss(this_cos_pred, GT[b, 7][this_nonoverlap_obj_mask], reduction='mean')
                obj_sin_loss += smooth_l1loss(this_sin_pred, GT[b, 8][this_nonoverlap_obj_mask], reduction='mean')
                obj_th_loss += smooth_l1loss(this_th_pred, GT[b, 9][this_nonoverlap_obj_mask], reduction='mean')
                obj_tw_loss += smooth_l1loss(this_tw_pred, GT[b, 10][this_nonoverlap_obj_mask], reduction='mean')
                
                total_nonoverlap_objs += 1
    
    # Average among contacing hands
    if total_nonoverlap_contact_hands > 0:
        tr_loss = tr_loss / total_nonoverlap_contact_hands
        cos_loss = cos_loss / total_nonoverlap_contact_hands
        sin_loss = sin_loss / total_nonoverlap_contact_hands
        th_loss = th_loss / total_nonoverlap_contact_hands
        tw_loss = tw_loss / total_nonoverlap_contact_hands
    else:
        tr_loss = torch.tensor(0.).to(device)
        cos_loss = torch.tensor(0.).to(device)
        sin_loss = torch.tensor(0.).to(device)
        th_loss = torch.tensor(0.).to(device)
        tw_loss = torch.tensor(0.).to(device)
    
    # Avearge among hands
    if total_nonoverlap_hands > 0:
        contact_loss = contact_loss / total_nonoverlap_hands
    else:
        contact_loss = torch.tensor(0.).to(device)
    
    # Average among objs
    if total_nonoverlap_objs > 0:
        obj_tr_loss = obj_tr_loss / total_nonoverlap_objs
        obj_cos_loss = obj_cos_loss / total_nonoverlap_objs
        obj_sin_loss = obj_sin_loss / total_nonoverlap_objs
        obj_th_loss = obj_th_loss / total_nonoverlap_objs
        obj_tw_loss = obj_tw_loss / total_nonoverlap_objs
    else:
        obj_tr_loss = torch.tensor(0.).to(device)
        obj_cos_loss = torch.tensor(0.).to(device)
        obj_sin_loss = torch.tensor(0.).to(device)
        obj_th_loss = torch.tensor(0.).to(device)
        obj_tw_loss = torch.tensor(0.).to(device)

    total_loss = tr_loss + cos_loss + sin_loss + th_loss + tw_loss + \
        contact_loss + obj_tr_loss + obj_cos_loss + \
        obj_sin_loss + obj_th_loss + obj_tw_loss + obj_prob_loss

    losses = {
        'tr_loss': tr_loss,
        'cos_loss': cos_loss,
        'sin_loss': sin_loss,
        'th_loss': th_loss,
        'tw_loss': tw_loss,
        'obj_tr_loss': obj_tr_loss,
        'obj_cos_loss': obj_cos_loss,
        'obj_sin_loss': obj_sin_loss,
        'obj_th_loss': obj_th_loss,
        'obj_tw_loss': obj_tw_loss,
        'contact_loss': contact_loss,
        'obj_prob_loss': obj_prob_loss,
        'total_loss': total_loss
    }
    return losses
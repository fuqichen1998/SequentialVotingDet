import numpy as np
from utils import *
from model.metric import get_AP_HO

# read gts and image sizes
data = read_pickle('saved/cache/test_gt_handobj_bboxes.pkl')
filenames = data['filenames']
gt_hand_boxes = data['hand_boxes']
gt_obj_boxes = data['obj_boxes']
hws = read_pickle('saved/cache/test_imgsize.pkl')
num_imgs = len(filenames)

# read and parse doh100 hand detection results
data = read_pickle('saved/cache/100doh_hand_det_res_formatted.pkl')
doh100_hand_boxes = data['hand_boxes']
doh100_hand_scores = data['hand_scores']

data_path = 'saved/outputs/exp_doh100/pred_epoch=5_use_gt_hands=False.pickle'
print("Evaluating {}".format(data_path))
score_thres = 0.
data = read_pickle(data_path)
det_thres, contact_thres = 0.2, 0.1
pred_obj_boxes = []
pred_obj_scores = []
pred_confidence_scores = []

for i, (hand_box, hand_score, pred_obj) in enumerate(zip(doh100_hand_boxes, doh100_hand_scores, data['dets'])):
    obj_box = pred_obj[:, :4] / 512. * max(hws[i])
    contact_score = pred_obj[:, 4]
    obj_score = pred_obj[:, 5] if pred_obj.shape[1] > 5 else np.ones_like(contact_score)
    # remove obj boxs
    obj_box[contact_score * obj_score < det_thres] = 0
    pred_obj_boxes.append(obj_box)
    pred_obj_scores.append(obj_score)
    # confidence score calculation
    mask = contact_score > contact_thres
    confidence_score = (hand_score * (1-contact_score)) * (~mask) + (hand_score * contact_score * obj_score) * mask
    pred_confidence_scores.append(confidence_score)

for iou_thres in [0.75, 0.5, 0.25]:
    prec, rec, ap = get_AP_HO(doh100_hand_boxes, doh100_hand_scores, pred_obj_boxes, pred_obj_scores, pred_confidence_scores, 
                            gt_hand_boxes, gt_obj_boxes, iou_thres=iou_thres)
    print('AP(IoU >{:.2f}): {:.4f}'.format(iou_thres, ap))

from utils import *
import torchvision
from model.metric import get_AP_


# read gts and image sizes
obj_data = read_pickle('saved/cache/test_gt_obj_bboxes_mcn.pkl')
heights = obj_data['heights']
widths = obj_data['widths']

data_path = 'saved/outputs/exp_meccano/pred_epoch=5_use_gt_hands=True.pickle'
print("Evaluating {}".format(data_path))

data = read_pickle(data_path)
dets = data['dets']
pred_obj_boxes = []
pred_obj_scores = []
for i, pred_obj in enumerate(dets):
    obj_box = pred_obj[:, :4] / 512. * max(heights[i], widths[i])
    obj_score = pred_obj[:, 5] if pred_obj.shape[1] > 5 else pred_obj[:, 4]
    pred_obj_boxes.append(obj_box)
    pred_obj_scores.append(obj_score)


# active object only
pred_obj_boxes_aft_nms = []
pred_obj_scores_aft_nms = []
for obj_boxs, obj_scores in zip(pred_obj_boxes, pred_obj_scores):
    idxs = torchvision.ops.nms(torch.tensor(obj_boxs), torch.tensor(obj_scores), 0.3)
    pred_obj_boxes_aft_nms.append(obj_boxs[idxs])
    pred_obj_scores_aft_nms.append((obj_scores)[idxs])

for iou_thres in [0.25, 0.5, 0.75]:
    prec, rec, ap, correct_det_indexs = get_AP_(
        pred_obj_boxes_aft_nms, pred_obj_scores_aft_nms, obj_data['obj_boxes'], iou_thres=iou_thres)
    print('AP(IoU >{:.2f}): {:.4f}'.format(iou_thres, ap))
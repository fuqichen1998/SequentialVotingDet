import json
import torch
import random
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image
from base import BaseDataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from .data_utils import threshold, cropped_resized_x, enlarge_box


class MCNDataLoader(BaseDataLoader):
    """
    MECCANO image dataloader
    """
    
    def __init__(self, 
                 batch_size, 
                 shuffle=True, 
                 validation_split=0.0, 
                 num_workers=1, 
                 **kwargs
                 ):
        self.dataset = MCNDataset(**kwargs)
        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers)



class MCNDataset(Dataset):
    """
    MECCANO image dataset
    """
    
    def __init__(self,
                 priors,
                 img_path_tp='MECCANO_FOLDER/active_object_frames/{}',
                 annotation_file_tp='MECCANO_FOLDER/home/fragusa/meccano_handobj_{}.json',
                 mode='train',
                 inp_size=224,
                 bbox_area_thres=1,
                 bbox_w_thres=1,
                 bbox_h_thres=1,
                 hand_bbox_ratio=1.,
                 max_nhands = 3,
                 max_nobjs = 3,
                 use_gt_hand = True,
                 ):

        self.img_path_tp = img_path_tp
        with open(annotation_file_tp.format(mode), 'r') as j:
            self.annotations = json.loads(j.read())
        self.keyframe_list = sorted(self.annotations.keys())

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01)

        # set properties
        self.mode = mode
        self.priors = priors
        self.inp_size = inp_size
        self.len = len(self.keyframe_list)
        self.hand_bbox_ratio = hand_bbox_ratio
        self.use_gt_hand = True
        self.max_nhands = max_nhands
        self.max_nobjs = max_nobjs
        
        # threshold for minimal bbox
        self.bbox_area_thres = bbox_area_thres
        self.bbox_h_thres = bbox_h_thres
        self.bbox_w_thres = bbox_w_thres

    def hand_bbox_aug(self, hand_bbox, h_thres, w_thres):
        x1, y1, x2, y2 = hand_bbox
        dh = (y2-y1)*(self.hand_bbox_ratio-1)//2
        dw = (x2-x1)*(self.hand_bbox_ratio-1)//2
        x1 = threshold(x1-dw, 0, w_thres)
        y1 = threshold(y1-dh, 0, h_thres)
        x2 = threshold(x2+dw, 0, w_thres)
        y2 = threshold(y2+dh, 0, h_thres)
        return [x1, y1, x2, y2]
    
    def get_handobj_bboxs(self, handobj_info, crop_info):
        trans_fn = crop_info['trans_fn']
        h = crop_info['h']
        w = crop_info['w']
        ci = crop_info['ci']
        cj = crop_info['cj']
        rh = crop_info['rh']
        rw = crop_info['rw']
        h_thres = crop_info['h_thres']
        w_thres = crop_info['w_thres']
        horizental_flip = crop_info['horizental_flip']
        vertical_flip = crop_info['vertical_flip']
        
        nhands = len(handobj_info)
        hand_bboxs = np.zeros((nhands, 4))
        obj_bboxs = np.zeros((nhands, 4))
        for i, handobj in enumerate(handobj_info):
            x1 = trans_fn(handobj['x1'], w, cj, rw, 0, w_thres)
            y1 = trans_fn(handobj['y1'], h, ci, rh, 0, h_thres)
            x2 = trans_fn(handobj['x2'], w, cj, rw, 0, w_thres)
            y2 = trans_fn(handobj['y2'], h, ci, rh, 0, h_thres)
            # augment hand bbox
            hand_bboxs[i] = self.hand_bbox_aug([x1, y1, x2, y2], h_thres, w_thres)
            if handobj.get("obj_bbox", None) is not None:
                bbox_info = handobj['obj_bbox']
                
                bx1 = trans_fn(bbox_info['x1'], w, cj, rw, 0, w_thres)
                by1 = trans_fn(bbox_info['y1'], h, ci, rh, 0, h_thres)
                bx2 = trans_fn(bbox_info['x2'], w, cj, rw, 0, w_thres)
                by2 = trans_fn(bbox_info['y2'], h, ci, rh, 0, h_thres)
                obj_bboxs[i] = [bx1, by1, bx2, by2]

        # flip boxes
        if horizental_flip:
            hand_bboxs[:, 0], hand_bboxs[:, 2] = w_thres - \
                hand_bboxs[:, 2], w_thres - hand_bboxs[:, 0]
            obj_bboxs[:, 0], obj_bboxs[:, 2] = w_thres - \
                obj_bboxs[:, 2], w_thres - obj_bboxs[:, 0]
        if vertical_flip:
            hand_bboxs[:, 1], hand_bboxs[:, 3] = h_thres - \
                hand_bboxs[:, 3], h_thres - hand_bboxs[:, 1]
            obj_bboxs[:, 1], obj_bboxs[:, 3] = h_thres - \
                obj_bboxs[:, 3], h_thres - obj_bboxs[:, 1]
        
        # round to int
        hand_bboxs = np.rint(hand_bboxs).astype(np.int)
        obj_bboxs = np.rint(obj_bboxs).astype(np.int)
        return hand_bboxs, obj_bboxs

    def prepare_GT(self, hand_bboxs, obj_bboxs, crop_info):
        # priors
        h_a = self.priors['h_a']
        w_a = self.priors['w_a']
        r_a = self.priors['r_a']
        obj_r_a = self.priors['obj_r_a']
        h_thres = crop_info['h_thres']
        w_thres = crop_info['w_thres']

        # 6 for tr, a, b, tw, th, contact(1 if there is contact else 0)
        # extra 6 for obj_tr, obj_a, obj_b, obj_tw, obj_th, obj_mask
        h, w = self.inp_size, self.inp_size
        GT = torch.zeros((12, h, w))
        handmask = torch.zeros((h, w))
        objbox_mask = torch.zeros((h, w))

        padded_enlarged_unqiue_hand_bboxes = np.zeros((self.max_nhands, 4)).astype(np.int)
        padded_enlarged_obj_bboxes = np.zeros((self.max_nhands, 4)).astype(np.int)
        for i, hand_bbox in enumerate(hand_bboxs):
            # handmaks = number of hands in a pixel
            x1, y1, x2, y2 = hand_bbox
            handmask[y1:y2, x1:x2] += 1.
            padded_enlarged_unqiue_hand_bboxes[i] = [x1, y1, x2, y2]

            if y1 < y2 and x1 < x2:
                bx1, by1, bx2, by2 = obj_bboxs[i]
                bbox_cy = (by1+by2) / 2.
                bbox_cx = (bx1+bx2) / 2.
                # hand
                hand_bbox_ys, hand_bbox_xs = torch.meshgrid(
                    torch.arange(y1, y2), torch.arange(x1, x2))
                rs = np.sqrt((hand_bbox_ys-bbox_cy)**2 + (hand_bbox_xs-bbox_cx)**2)+1e-7
                if by1 < by2 and bx1 < bx2:
                    padded_enlarged_obj_bboxes[i] = [bbox_cx, bbox_cy, by2-by1, bx2-bx1]
                    # tr
                    GT[0, y1:y2, x1:x2] = torch.clip(torch.log(rs/r_a), -6, 2)
                    # hand cos, sin
                    GT[1, y1:y2, x1:x2] = (hand_bbox_xs-bbox_cx) / rs
                    GT[2, y1:y2, x1:x2] = (hand_bbox_ys-bbox_cy) / rs
                    # hand th, tw
                    GT[3, y1:y2, x1:x2] = torch.log(torch.tensor((by2-by1)/h_a))
                    GT[4, y1:y2, x1:x2] = torch.log(torch.tensor((bx2-bx1)/w_a))
                    # contact
                    GT[5, y1:y2, x1:x2] = 1.

        padded_enlarged_unqiue_obj_bboxes = np.zeros((self.max_nobjs, 4)).astype(np.int)
        unqiue_obj_bboxes = np.unique(obj_bboxs, axis=0)
        for i, obj_bbox in enumerate(unqiue_obj_bboxes):
            bx1, by1, bx2, by2 = obj_bbox
            h, w = by2-by1, bx2-bx1
            # enlarge the obj bbox
            elged_bx1, elged_by1, elged_bx2, elged_by2 = np.rint(enlarge_box(obj_bbox, h_thres, w_thres, 1.4)).astype(np.int)
            padded_enlarged_unqiue_obj_bboxes[i] = [elged_bx1, elged_by1, elged_bx2, elged_by2]
            
            objbox_mask[elged_by1:elged_by2, elged_bx1:elged_bx2] += 1.
            bbox_cy = (by1+by2) / 2.
            bbox_cx = (bx1+bx2) / 2.
            # object
            obj_bbox_ys, obj_bbox_xs = torch.meshgrid(torch.arange(elged_by1, elged_by2), torch.arange(elged_bx1, elged_bx2))
            obj_rs = np.sqrt((obj_bbox_ys-bbox_cy)**2 + (obj_bbox_xs-bbox_cx)**2)+1e-7
            if by1 < by2 and bx1 < bx2:
                # obj_tr
                GT[6, elged_by1:elged_by2, elged_bx1:elged_bx2] = torch.clip(torch.log(obj_rs/obj_r_a), -6, 2)
                # obj cos, sin
                GT[7, elged_by1:elged_by2, elged_bx1:elged_bx2] = (obj_bbox_xs-bbox_cx) / obj_rs
                GT[8, elged_by1:elged_by2, elged_bx1:elged_bx2] = (obj_bbox_ys-bbox_cy) / obj_rs
                # obj th, tw
                GT[9, elged_by1:elged_by2, elged_bx1:elged_bx2] = torch.log(torch.tensor(h/h_a))
                GT[10, elged_by1:elged_by2, elged_bx1:elged_bx2] = torch.log(torch.tensor(w/w_a))
                GT[11, by1:by2, bx1:bx2] = 1.

        # only train contact on nonoverlaped hand mask
        handmask = handmask == 1.

        # only train tr, a, b, tw, th on nonoverlaped hand contact with objs
        contact_mask = handmask * GT[5, :, :]
        
        # only train obj_tr, obj_a, obj_b, obj_tw, obj_th on nonoverlaped object
        objbox_mask = objbox_mask == 1.

        return {
            'GT': GT, 
            'handmask': handmask, 
            'contact_mask': contact_mask, 
            'objbox_mask': objbox_mask, 
            'nhands': hand_bboxs.shape[0],
            'n_unique_objs': unqiue_obj_bboxes.shape[0],
            'padded_enlarged_unqiue_hand_bboxes': padded_enlarged_unqiue_hand_bboxes,
            'padded_enlarged_obj_bboxes': padded_enlarged_obj_bboxes,
            'padded_enlarged_unqiue_obj_bboxes': padded_enlarged_unqiue_obj_bboxes,
            # priors
            'h_a': h_a,
            'w_a': w_a,
            'r_a': r_a,
            'obj_r_a': obj_r_a,
            }

    def train_data_augmentation(self, keyframe):
        h, w = keyframe.shape[1], keyframe.shape[2]
        
        # Randomly crop with a random aspect ratio = w/h
        ch = int(random.randint(int(0.8*h), h))
        cw = int(random.randint(int(0.8*w), w))
        ci, cj, ch, cw = transforms.RandomCrop.get_params(keyframe, (ch, cw))
        r_h, r_w = self.inp_size / max(ch, cw), self.inp_size / max(ch, cw)
        keyframe = F.crop(keyframe, ci, cj, ch, cw)
        
        # random horizental flip
        horizental_flip = False
        if random.random() > 0.5:
            horizental_flip = True
            keyframe = keyframe.flip(2)
        
        # no random vertical flip
        vertical_flip = False
        # if random.random() > 0.5:
        #     vertical_flip = True
        #     keyframe = keyframe.flip(1)
        
        # color jitter
        keyframe = self.color_jitter(keyframe)

        # 6. pad image to square
        padding = (0, 0, max(ch, cw) - cw, max(ch, cw) - ch)
        keyframe = F.pad(keyframe, padding)
        
        # resize the image
        keyframe = transforms.Resize(self.inp_size)(keyframe)

        crop_info = {
            'trans_fn': cropped_resized_x,
            'h': h,
            'w': w,
            'ci': ci,
            'cj': cj,
            'rh': r_h,
            'rw': r_w,
            'h_thres': ch * r_h,
            'w_thres': cw * r_w,
            'horizental_flip': horizental_flip,
            'vertical_flip': vertical_flip
        }
        return keyframe, crop_info


    def eval_data_prep(self, keyframe):
        h, w = keyframe.shape[1], keyframe.shape[2]
        
        # recompute the index of the bounding boxes
        r_h, r_w = self.inp_size / max(h, w), self.inp_size / max(h, w)
        
        # pad image to square
        padding = (0, 0, max(h, w) - w, max(h, w) - h)
        keyframe = F.pad(keyframe, padding)
        
        # resize the image, GT, handmask, contact_mask
        keyframe = transforms.Resize(self.inp_size)(keyframe)

        crop_info = {
            'trans_fn': cropped_resized_x,
            'h': h,
            'w': w,
            'ci': 0,
            'cj': 0,
            'rh': r_h,
            'rw': r_w,
            'h_thres': h * r_h,
            'w_thres': w * r_w,
            'horizental_flip': False,
            'vertical_flip': False
        }
        return keyframe, crop_info

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # get image meta
        keyframe_subpath = self.keyframe_list[idx]
        annotations = self.annotations[keyframe_subpath]
        
        # load keyframe
        keyframe_path = self.img_path_tp.format(keyframe_subpath)
        keyframe = transforms.ToTensor()(Image.open(keyframe_path))
        oh, ow = keyframe.shape[1], keyframe.shape[2]
       
        # augmentation
        if 'train' in self.mode:
            # train
            keyframe, crop_info = self.train_data_augmentation(keyframe)
        else:
            # eval
            keyframe, crop_info = self.eval_data_prep(keyframe) 
        
        # normalize
        keyframe = self.normalize(keyframe)
        
        # compute GT
        hand_bboxs, obj_bboxs = self.get_handobj_bboxs(annotations, crop_info)
        label = self.prepare_GT(hand_bboxs, obj_bboxs, crop_info)

        # return bboxs for test
        if self.mode == 'test':
            label['hand_bboxs'] = hand_bboxs
            label['obj_bboxs'] = obj_bboxs
            label['h'] = oh
            label['w'] = ow
            
        return keyframe, label

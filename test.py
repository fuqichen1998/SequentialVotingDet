import torch
import argparse
import collections
import model.model as module_arch
import data_loader.data_loaders as module_data

from utils import *
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from parse_config import ConfigParser
from model.voting import get_obj_predictions_iterative


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('test_data_loader', module_data)
    use_gt_hand = data_loader.dataset.use_gt_hand
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    load_epoch = checkpoint['epoch']
    state_dict = checkpoint['state_dict']
    # if previous model is in data parallel
    new_model_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k[7:]
        new_model_state_dict[k] = v
    state_dict = new_model_state_dict
    
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # if the current model in data parallel
    if type(model) == torch.nn.DataParallel:
        new_model_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.'+k
            new_model_state_dict[k] = v
        state_dict = new_model_state_dict
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # test dataloader must have a batch size of 1
    assert(data_loader.batch_size == 1)
    dets = []
    obj_bboxs = []
    hand_bboxs = []
    priors = data_loader.dataset.priors
    
    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(data_loader)):
            
            GT, handmask, objmask, objbox_mask = target['GT'], target[
                'handmask'], target['contact_mask'], target['objbox_mask']
            hand_bbox, obj_bbox = target['hand_bboxs'], target['obj_bboxs']
            data, GT, handmask, objmask, objbox_mask = data.to(device), GT.to(
                device), handmask.to(device), objmask.to(device), objbox_mask.to(device)
            output = model(data)        
            # iterative votings
            if use_gt_hand:
                rough_obj_det, det = get_obj_predictions_iterative(output[0], handmask[0], hand_bbox[0], priors)
            else:
                rough_obj_det, det = get_obj_predictions_iterative(output[0], handmask[0], target['doh100_hand_box'][0], priors)

            dets.append(det)
            hand_bboxs.append(deepcopy(hand_bbox[0].detach().cpu().numpy()))
            obj_bboxs.append(deepcopy(obj_bbox[0].detach().cpu().numpy()))
    
    save_dir = Path(config['trainer']['save_dir'])
    output_dir = save_dir / 'outputs' / config['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save output
    res = {'dets': dets, 'hand_bboxs': hand_bboxs}
    fname = 'pred_epoch={}_use_gt_hands={}.pickle'.format(load_epoch, use_gt_hand)
    write_pickle(res, output_dir / fname)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--rid', '--run_id'], type=str, target='run_id'),
        CustomArgs(['--ngpu', '--num_gpus'], type=int, target='n_gpu'),
        CustomArgs(['--hbr', '--hand_bbox_ratio'], type=float, target='test_data_loader;args;hand_bbox_ratio'),
        CustomArgs(['--use_gt_hand'], type=bool, target='test_data_loader;args;use_gt_hand'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

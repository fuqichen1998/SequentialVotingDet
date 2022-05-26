import torch
import argparse
import collections
import model.loss as module_loss
import model.model as module_arch
import model.metric as module_metric
import data_loader.data_loaders as module_data

from trainer import Trainer
from utils import prepare_device
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.init_obj('train_data_loader', module_data)
    valid_data_loader = config.init_obj('val_data_loader', module_data)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    
    # load weights from pretraining
    if 'pretrained_net' in config.config:
        state_dict = torch.load(config['pretrained_net'])['state_dict']
        model.encoder.load_state_dict(state_dict)
        logger.info("loaded pretrained weight from {}.".format(config['pretrained_net']))
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.transformer_encoder.parameters():
            param.requires_grad = False
        for param in model.decoder.parameters():
            param.requires_grad = False
        logger.info("Encoder Attention Decoder weights freezed.")

    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=None, # skip validation to save time
                      lr_scheduler=lr_scheduler)

    trainer.train()


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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='train_data_loader;args;batch_size'),
    ]
    config = ConfigParser.from_args(args, options)
    
    main(config)

import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import pathlib

import os 
from multiprocessing import cpu_count
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.set_printoptions(precision=8)
#torch.set_default_tensor_type(torch.DoubleTensor)

pathlib.PosixPath = pathlib.WindowsPath

# cpu_num = cpu_count()
# os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
# os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
# os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
# os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
# os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# torch.set_num_threads(cpu_num)
# torch.autograd.set_detect_anomaly(True)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    train_data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = config.init_obj('valid_data_loader', module_data)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch, device=device)
    logger.info(model)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    train_epochs = config["trainer"]["epochs"]
    criterion = config.init_obj('loss', module_loss, epochs=train_epochs, device=device)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r',
                      '--resume',
                      default=None,
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-i',
                      '--init_model',
                      default=None,
                      type=str,
                      help='path to select checkpoint (default: None)')
    args.add_argument('-d',
                      '--device',
                      default=None,
                      type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [CustomArgs(['--lr', '--learning_rate'],type=float,target='optimizer;args;lr'),
               CustomArgs(['--bs', '--batch_size'],type=int,target='data_loader;args;batch_size')]
    config = ConfigParser.from_args(args, options)
    main(config)

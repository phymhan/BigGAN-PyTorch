""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import functools
import math
import numpy as np
from tqdm import tqdm, trange


import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
import torchvision

# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
import copy

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  # config['resolution'] = utils.imsize_dict[config['dataset']]
  config['resolution'] = utils.imsize_dict.get(config['dataset'], config['image_size'])
  config['n_classes'] = utils.nclass_dict.get(config['dataset'], config['num_classes'])
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
    config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda'
  
  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # load Inception model
  model = inception_utils.inception_v3_with_custom_num_classes(config['n_classes'])
  model = model.to(device)
  print(model)

  if config['optimizer'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  elif config['optimizer'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0., 0.999), weight_decay=0, eps=1e-8)

  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'config': config}

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  if not os.path.exists(os.path.join(config['weights_root'], experiment_name)):
    os.makedirs(os.path.join(config['weights_root'], experiment_name))
  
  # set tensorboard logger
  tb_writer = None
  if config['tensorboard']:
    from torch.utils.tensorboard import SummaryWriter
    tb_logdir = '%s/%s/tblogs' % (config['logs_root'], experiment_name)
    if os.path.exists(tb_logdir) and not config['resume']:
      for filename in os.listdir(tb_logdir):
        if filename.startswith('events'):
          os.remove(os.path.join(tb_logdir, filename))  # remove previous event logs
    tb_writer = SummaryWriter(log_dir=tb_logdir)
  # Write metadata
  utils.write_metadata(config['logs_root'], experiment_name, config, state_dict)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  config['num_D_accumulations'] = 1
  config['num_D_steps'] = 1
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})
  train = train_fns.inception_training_function(model, optimizer, state_dict, config)
  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Train for specified number of epochs, although we mostly track G iterations.
  for epoch in range(state_dict['epoch'], config['num_epochs']):    
    # Which progressbar to use? TQDM or my own?
    if config['pbar'] == 'mine':
      pbar = utils.progress(loaders[0], displaytype='s1k' if config['use_multiepoch_sampler'] else 'eta')
    else:
      pbar = tqdm(loaders[0])
    for i, (x, y) in enumerate(pbar):
      # Increment the iteration counter
      state_dict['itr'] += 1
      # Make sure G and D are in training mode, just in case they got set to eval
      # For D, which typically doesn't have BN, this shouldn't matter much.
      model.train()
      x, y = x.to(device), y.to(device)
      metrics = train(x, y)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      if config['tensorboard']:
        for metric_name in metrics:
          tb_writer.add_scalar('Train/%s' % metric_name, metrics[metric_name], state_dict['itr'])

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        model_weights = copy.deepcopy(model.state_dict())
        torch.save(model_weights, os.path.join(config['weights_root'], experiment_name, 'model_itr_%d.pth'%state_dict['itr']))
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  # print(config)
  utils.print_config(parser, config)
  run(config)

if __name__ == '__main__':
  main()
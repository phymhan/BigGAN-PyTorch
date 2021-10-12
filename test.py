""" BigGAN: The Authorized Unofficial PyTorch release
    Code by A. Brock and A. Andonian
    This code is an unofficial reimplementation of
    "Large-Scale GAN Training for High Fidelity Natural Image Synthesis,"
    by A. Brock, J. Donahue, and K. Simonyan (arXiv 1809.11096).

    Let's go.
"""

import os
import shutil
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


def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict.get(config['dataset'], config['image_size'])
  config['n_classes'] = utils.nclass_dict.get(config['dataset'], config['num_classes'])
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['skip_init'] = True
  config = utils.update_config_roots(config)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Seed RNG
  utils.seed_rng(config['seed'])

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)

  # Next, build the model
  G = model.Generator(**config).to(device)
  
   # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
  else:
    G_ema = None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  print(G)
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0,
                'best_FID': 999999, 'best_meanFID': 999999, 'best_maxFID': 999999,
                'best_meanLPIPS': -1, 'best_minLPIPS': -1, 'config': config}
  
  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  # test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
  #                                           experiment_name)
  test_metrics_fname = '%s/%s_%s.txt' % (config['logs_root'], experiment_name, config['test_prefix'])
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, reinitialize=False)

  # Prepare inception metrics: FID and IS
  intra_fid_classes = utils.str2list(config['intra_fid_classes']) if config['intra_fid_classes'] else list(range(config['n_classes']))
  print(f'Intra-FID will be calculated for classes {intra_fid_classes} (for all classes if empty).')
  lpips_classes = utils.str2list(config['lpips_classes']) if config['lpips_classes'] else list(range(config['n_classes']))
  print(f'LPIPS will be calculated for classes {lpips_classes} (for all classes if empty).')
  get_inception_metrics = inception_utils.prepare_inception_metrics(config['dataset'], config['data_root'], config['parallel'],
                                                                    config['no_fid'], config['use_torch'],
                                                                    config['no_intra_fid'], intra_fid_classes, config['load_single_inception_moments'],
                                                                    custom_inception_model_path=config['custom_inception_model_path'],
                                                                    custom_num_classes=config['custom_num_classes'],
                                                                    use_lpips=config['use_lpips'], lpips_classes=lpips_classes, config=config)

  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  # Prepare a fixed z & y to see individual sample evolution throghout training
  fixed_z, fixed_y = utils.prepare_z_y(G_batch_size, G.dim_z,
                                       config['n_classes'], device=device,
                                       fp16=config['G_fp16'])  
  fixed_z.sample_()
  fixed_y.sample_()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                             G=(G_ema if config['ema'] and config['use_ema'] else G),
                             z_=z_, y_=y_, config=config)
  # which_metrics = ['IS']
  # if not config['no_fid']:
  #   which_metrics += ['FID']
  # if not config['no_intra_fid']:
  #   which_metrics += ['IntraFID']
  # if config['use_lpips']:
  #   which_metrics += ['LPIPS']
  which_metrics = config['which_metrics']
  for name_suffix in config['load_weights']:
    print(f'Loading weights {name_suffix}...')
    utils.load_weights(G, None, state_dict,
                       config['weights_root'], experiment_name, 
                       name_suffix, G_ema if config['ema'] else None)
    # Sample
    if config['G_eval_mode']:
      G.eval()
      if config['ema']:
        G_ema.eval()
    train_fns.test_sample(G, None, G_ema, z_, y_, fixed_z, fixed_y, 
                          state_dict, config, experiment_name,
                          config['test_prefix'], name_suffix)
    # Test
    if config['G_eval_mode']:
      G.eval()
    train_fns.test_metric(G, None, G_ema, z_, y_, state_dict, config, sample,
                          get_inception_metrics, experiment_name, test_log,
                          '%s/%s' % (config['logs_root'], experiment_name),
                          config['test_prefix'], name_suffix,
                          which_metrics=which_metrics)


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  run(config)

if __name__ == '__main__':
  main()
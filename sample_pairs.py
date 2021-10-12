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
  config['ema'] = config['use_ema'] = False
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
  G_ema = None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
  
  print(G)
  # Prepare state dict, which holds things like epoch # and itr #

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
  # sample = functools.partial(utils.sample,
  #                            G=(G_ema if config['ema'] and config['use_ema'] else G),
  #                            z_=z_, y_=y_, config=config)
  assert(os.path.isfile(config['test_G_model_path']))
  G.load_state_dict(torch.load(config['test_G_model_path']), strict=True)

  # Sample
  if config['G_eval_mode']:
    G.eval()
  utils.sample_pairs(G,
                     G_batch_size=G_batch_size,
                     samples_per_class=config['samples_per_class'],
                     parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=f'pairs',
                     z_=z_, y_=y_)


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  run(config)

if __name__ == '__main__':
  main()
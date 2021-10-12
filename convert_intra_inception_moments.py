''' Convert Intra Inception Moments
 Save a single file as separate files. '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import inception_utils
from tqdm import tqdm, trange
from argparse import ArgumentParser
import os

def prepare_parser():
  usage = 'Calculate and store inception metrics.'
  parser = ArgumentParser(description=usage)
  parser.add_argument('--custom_inception_model_path', type=str, default='', help='if not empty, load custom inception model')
  parser.add_argument('--custom_num_classes', type=int, default=2000, help='num classes')
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--out_root', type=str, default='data')
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  n_classes = utils.nclass_dict.get(config['dataset'], config['num_classes'])
  src_path = os.path.join(config['out_root'], config['dataset'].strip('_hdf5')+'_intra_inception_moments.npz')
  tar_path = os.path.join(config['out_root'], config['dataset'].strip('_hdf5')+'_intra_inception_moments')
  if not os.path.exists(tar_path):
    os.mkdir(tar_path)
  mu = np.load(src_path)['mu']
  sigma = np.load(src_path)['sigma']
  for y in range(n_classes):
    np.savez(os.path.join(tar_path, f'{y:04d}.npz'), **{'mu': mu[y], 'sigma': sigma[y]})
    print(f'moments for class {y} saved to {tar_path}/{y:04d}.npz')

def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()
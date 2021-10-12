''' Calculate Inception Moments
 This script iterates over the dataset and calculates the moments of the 
 activations of the Inception net (needed for FID), and also returns
 the Inception Score of the training data.
 
 Note that if you don't shuffle the data, the IS of true data will be under-
 estimated as it is label-ordered. By default, the data is not shuffled
 so as to reduce non-determinism. '''
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
  parser.add_argument('--use_custom_dataset', action='store_true')
  parser.add_argument('--image_size', type=int, default=64, help='only valid if use_custom_dataset')
  parser.add_argument('--num_classes', type=int, default=2, help='only valid if use_custom_dataset')
  parser.add_argument('--inception_moments_path', type=str, default='')
  parser.add_argument('--intra_inception_moments_path', type=str, default='')
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100...'
         'Append _hdf5 to use the hdf5 version of the dataset. (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--out_root', type=str, default='data')
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data? (default: %(default)s)') 
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use.')
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  loaders = utils.get_data_loaders(**config)
  nclass = utils.nclass_dict.get(config['dataset'], config['num_classes'])

  labels = []
  device = 'cuda'
  total_num = 0
  for i, (x, y) in enumerate(tqdm(loaders[0])):
    labels += [np.asarray(y.cpu())]
    total_num += y.size(0)

  labels = np.concatenate(labels, 0)

  # histogram
  count, _ = np.histogram(labels, nclass, [0, nclass])
  prior = np.maximum(count / count.sum(), 1e-7)
  log_prior = np.log(prior)

  # uncomment to save pool, logits, and labels to disk
  np.savez(os.path.join(config['out_root'], config['dataset'].strip('_hdf5')+'_labels.npz'),
           **{'labels': labels, 'prior': prior, 'log_prior': log_prior})
  print(f'total number of images: {total_num}')
  print(f'label distributions: {prior}')


def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
  main()

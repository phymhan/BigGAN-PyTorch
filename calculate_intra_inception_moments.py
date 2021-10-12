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
  parser.add_argument('--inception_model', type=str, default='default')
  parser.add_argument('--use_custom_dataset', action='store_true')
  parser.add_argument('--image_size', type=int, default=64, help='only valid if use_custom_dataset')
  parser.add_argument('--num_classes', type=int, default=2, help='only valid if use_custom_dataset')
  parser.add_argument('--index_filename', type=str, default='')
  parser.add_argument('--inception_moments_path', type=str, default='')
  parser.add_argument('--intra_inception_moments_path', type=str, default='')
  parser.add_argument('--load_single_inception_moments', action='store_true', help='load single file for mu and sigma?')
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
  n_classes = utils.nclass_dict.get(config['dataset'], config['num_classes'])

  # Load inception net
  if config['inception_model'] == 'default':
    if config['custom_inception_model_path']:
      print('loading custom inception model...')
      net = inception_utils.load_custom_inception_net(parallel=config['parallel'],
                                                      num_classes=config['custom_num_classes'],
                                                      model_path=config['custom_inception_model_path'])
    else:
      net = inception_utils.load_inception_net(parallel=config['parallel'])
  elif config['inception_model'] == 'studio':
    net = inception_utils.load_studio_inception_net(parallel=config['parallel'])
  pool, logits, labels = [[] for _ in range(n_classes)], [[] for _ in range(n_classes)], [[] for _ in range(n_classes)]
  device = 'cuda'
  for i, (x, y) in enumerate(tqdm(loaders[0])):
    x = x.to(device)
    with torch.no_grad():
      pool_val, logits_val = net(x)
      for j in range(y.size(0)):
        pool[y[j]] += [np.asarray(pool_val[j:j+1,...].cpu())]
        # logits[y[j]] += [np.asarray(F.softmax(logits_val[j:j+1,...], 1).cpu())]
        # labels[y[j]] += [np.asarray(y[j:j+1,...].cpu())]
  pool = [np.concatenate(item, 0) for item in pool]
  # logits = [np.concatenate(item, 0) for item in logits]
  # labels = [np.concatenate(item, 0) for item in labels]

  # Prepare mu and sigma, save to disk. Remove "hdf5" by default 
  # (the FID code also knows to strip "hdf5")
  print('Calculating means and covariances...')
  mu = [0 for _ in range(n_classes)]
  sigma = [0 for _ in range(n_classes)]
  tar_path = config['intra_inception_moments_path'] or os.path.join(config['out_root'], config['dataset'].strip('_hdf5')+'_intra_inception_moments')
  if not os.path.exists(tar_path):
    os.mkdir(tar_path)
  for y in range(n_classes):
    mu_, sigma_ = np.mean(pool[y], axis=0), np.cov(pool[y], rowvar=False)
    if config['load_single_inception_moments']:
      mu[y], sigma[y] = mu_, sigma_
    else:
      np.savez(os.path.join(tar_path, f'{y:04d}.npz'), **{'mu': mu_, 'sigma': sigma_})
      print(f'moments for class {y} saved to {tar_path}/{y:04d}.npz')
  if config['load_single_inception_moments']:
    print('Saving calculated means and covariances to disk...')
    filepath = config['intra_inception_moments_path'] or os.path.join(config['out_root'], config['dataset'].strip('_hdf5')+'_intra_inception_moments.npz')
    np.savez(filepath, **{'mu' : mu, 'sigma' : sigma})

def main():
  # parse command line    
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()
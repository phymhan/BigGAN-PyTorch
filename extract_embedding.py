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
from torch.utils import data
# Import my stuff
import inception_utils
import utils
import losses
import train_fns
from sync_batchnorm import patch_replication_callback
from natsort import natsorted
import random
import pickle
import pdb
st = pdb.set_trace

class MySampler(torch.utils.data.Sampler):
  def __init__(self, data_source, classes=range(10), samples_per_class=200, batch_size=128, index_filename='data_index.pkl'):
    self.data_source = data_source
    self.samples_per_class = samples_per_class
    self.num_samples = len(self.data_source)
    self.batch_size = batch_size
    if os.path.exists(index_filename):
      with open(index_filename, 'rb') as f:
        index, output = pickle.load(f)
    else:
      count = {y: 0 for y in classes}
      index = {y: [] for y in classes}
      output = []
      torch.manual_seed(0)
      j = torch.randperm(len(data_source)).tolist()
      for i in tqdm(j):
        _, y = data_source[i]
        if y not in classes or count[y] >= samples_per_class:
          continue
        count[y] += 1
        index[y].append(i)
        output.append(i)
        if sum(count.values()) >= samples_per_class * len(classes):
          break
      with open(index_filename, 'wb') as f:
        pickle.dump((index, output), f)
    self.classes = classes
    self.index = index
    self.output = output

  def __iter__(self):
    # output = []
    # for y in self.classes:
    #   output += self.index[y]
    # return iter(output)
    return iter(self.output)

  def __len__(self):
    return sum(self.count.values())


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
  seed = config['seed']
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Prepare root folders if necessary
  utils.prepare_root(config)

  # Setup cudnn.benchmark for free speed
  # torch.backends.cudnn.benchmark = True

  # Import the model--this line allows us to dynamically select different files.
  model = __import__(config['model'])
  experiment_name = (config['experiment_name'] if config['experiment_name']
                       else utils.name_from_config(config))
  print('Experiment name is %s' % experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  if not os.path.exists(os.path.join(train_metrics_fname, 'embedding')):
    os.mkdir(os.path.join(train_metrics_fname, 'embedding'))

  # Next, build the model
  G = model.Generator(**config).to(device)
  D = model.Discriminator(**config).to(device)
  
  # If using EMA, prepare it
  G_ema = None
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  # G_ema.eval()
  # D.eval()
  GD = model.G_D(G_ema, D, **config)

  # Prepare state dict, which holds things like epoch # and itr #
  # state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0,
  #               'best_FID': 999999, 'best_meanFID': 999999, 'best_maxFID': 999999,
  #               'best_meanLPIPS': -1, 'best_minLPIPS': -1, 'config': config}
  
  # Prepare inception metrics: FID and IS
  intra_fid_classes = utils.str2list(config['intra_fid_classes']) if config['intra_fid_classes'] else list(range(config['n_classes']))
  print(f'Intra-FID will be calculated for classes {intra_fid_classes} (for all classes if empty).')
  
  # Prepare noise and randomly sampled label arrays
  # Allow for different batch sizes in G
  G_batch_size = max(config['G_batch_size'], config['batch_size'])
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  dset = utils.get_data_loaders(**{**config, 'batch_size': G_batch_size, 'return_dataset': True})
  sampler = MySampler(dset, intra_fid_classes, config['samples_per_class'], index_filename=f"{config['dataset']}_index.pkl")
  loader = data.DataLoader(dset, batch_size=G_batch_size, sampler=sampler)
  # loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
  #                                     'start_itr': 0})
  z_batches = []
  y_batches = []
  z_, y_ = utils.prepare_z_y(G_batch_size, G.dim_z, config['n_classes'],
                             device=device, fp16=config['G_fp16'])
  for i in range(int(np.ceil(config['samples_per_class'] * len(intra_fid_classes) / G_batch_size * 1.0))):
    z_.sample_()
    # y_.sample_()
    z_batches.append(z_)
    # y_batches.append(y_)

  # Prepare Sample function for use with inception metrics
  # sample = functools.partial(utils.sample,
  #                            G=(G_ema if config['ema'] and config['use_ema'] else G),
  #                            z_=z_, y_=y_, config=config)
  weights_root = os.path.join(config['weights_root'], experiment_name)
  if len(config['load_weights']) == 0:
    load_weights = [s.replace('state_dict_', '').replace('.pth', '') for s in os.listdir(weights_root) if s.startswith('state_dict_')]
    load_weights = natsorted(load_weights)
  else:
    load_weights = config['load_weights']
  for name_suffix in load_weights:
    print(f'Loading weights {name_suffix}...')
    utils.load_weights(G, D, None,
                       config['weights_root'], experiment_name, 
                       name_suffix, G_ema)
    G_ema.eval()
    D.eval()
    embed_fake = []
    embed_real = []
    label_fake = []
    label_real = []
    epoch_name = name_suffix
    for i, (x, y) in enumerate(loader):
      batch_size = y.shape[0]
      x, y = x.to(device), y.to(device)
      with torch.no_grad():
        G_z = G_ema(z_batches[i][:batch_size], G_ema.shared(y))
        e_fake = D.get_embedding(G_z)
        e_real = D.get_embedding(x)
        embed_fake.append(e_fake.cpu())
        embed_real.append(e_real.cpu())
        label_fake.append(y.cpu())
        label_real.append(y.cpu())
        # label_fake.append(y_batches[i][:batch_size].cpu())
    embed_fake = torch.cat(embed_fake, 0)
    embed_real = torch.cat(embed_real, 0)
    label_real = torch.cat(label_real, 0)
    label_fake = torch.cat(label_fake, 0)
    np.save(os.path.join(train_metrics_fname, 'embedding', f"{epoch_name}_embed_real.npy"), embed_real.numpy())
    np.save(os.path.join(train_metrics_fname, 'embedding', f"{epoch_name}_embed_fake.npy"), embed_fake.numpy())
    np.save(os.path.join(train_metrics_fname, 'embedding', f"{epoch_name}_label_real.npy"), label_real.numpy())
    np.save(os.path.join(train_metrics_fname, 'embedding', f"{epoch_name}_label_fake.npy"), label_fake.numpy())


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  run(config)

if __name__ == '__main__':
  main()
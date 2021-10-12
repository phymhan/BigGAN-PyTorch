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

import pdb
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

# The main training file. Config is a dictionary specifying the configuration
# of this training run.
def run(config):

  # Update the config dict as necessary
  # This is for convenience, to add settings derived from the user-specified
  # configuration into the config-dict (e.g. inferring the number of classes
  # and size of the images from the dataset, passing in a pytorch object
  # for the activation specified as a string)
  config['resolution'] = utils.imsize_dict.get(config['dataset'], config['image_size'])
  config['n_classes'] = utils.nclass_dict.get(config['dataset'], config['num_classes'])
  config['G_activation'] = utils.activation_dict[config['G_nl']]
  config['D_activation'] = utils.activation_dict[config['D_nl']]
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
  D = model.Discriminator(**config).to(device)
  
  # If using EMA, prepare it
  if config['ema']:
    print('Preparing EMA for G with decay of {}'.format(config['ema_decay']))
    G_ema = model.Generator(**{**config, 'skip_init':True, 
                               'no_optim': True}).to(device)
    ema = utils.ema(G, G_ema, config['ema_decay'], config['ema_start'])
  else:
    G_ema, ema = None, None
  
  # FP16?
  if config['G_fp16']:
    print('Casting G to float16...')
    G = G.half()
    if config['ema']:
      G_ema = G_ema.half()
  if config['D_fp16']:
    print('Casting D to fp16...')
    D = D.half()
    # Consider automatically reducing SN_eps?
  GD = model.G_D(G, D, **config)
  # print(G)
  # print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0,
                'best_FID': 999999, 'best_meanFID': 99999, 'best_maxFID': 999999,
                'best_meanLPIPS': -1, 'best_minLPIPS': -1, 'config': config}
  M_state = {'M': 0, 'logit': 0}

  # If loading from a pre-trained model, load weights
  print('Loading weights...')
  utils.load_weights(G, D, None,
                      config['weights_root'], experiment_name, 
                      config['load_weights'][-1] if config['load_weights'] else None,
                      G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  test_metrics_fname = '%s/%s_%s.txt' % (config['logs_root'], experiment_name, config['test_prefix'])
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, reinitialize=False)
  # Prepare data; the Discriminator's batch size is all that needs to be passed
  # to the dataloader, as G doesn't require dataloading.
  # Note that at every loader iteration we pass in enough data to complete
  # a full D iteration (regardless of number of D steps and accumulations)
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})

  # Prepare inception metrics: FID and IS
  intra_fid_classes = utils.str2list(config['intra_fid_classes']) if config['intra_fid_classes'] else list(range(config['n_classes']))
  print(f'Intra-FID will be calculated for classes {intra_fid_classes} (for all classes if empty).')
  lpips_classes = utils.str2list(config['lpips_classes']) if config['lpips_classes'] else list(range(config['n_classes']))
  print(f'LPIPS will be calculated for classes {lpips_classes} (for all classes if empty).')
  drs_classes = utils.str2list(config['fair_drs_classes']) if config['fair_drs_classes'] else list(range(config['n_classes']))
  print(f'DRS will be performed for classes {drs_classes} (for all classes if empty).')
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
  # Loaders are loaded, prepare the training function
  train = train_fns.DRS_training_function(G, D, GD, z_, y_,
                                          ema, state_dict, config)
  
  print('Beginning training at epoch %d...' % state_dict['epoch'])
  # Discriminator fine-tuning
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
      if config['G_eval_mode']:
        print('Switchin G to eval mode...')
        G.eval()
        if config['ema']:
          G_ema.eval()
      else:
        G.train()
        if config['ema']:
          G_ema.train()
      D.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      metrics = train(x, y)
      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')
      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        utils.save_weights(None, D, state_dict, config['weights_root'],
                           experiment_name, 'drs_itr%d' %  state_dict['itr'],
                           G_ema if config['ema'] else None)
      if state_dict['itr'] >= config['max_iters_drs']:
        break
    # Increment epoch counter at end of epoch
    state_dict['epoch'] += 1
    if state_dict['itr'] >= config['max_iters_drs']:
      break
  
  if state_dict['itr'] == 0:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights_drs'][-1] if config['load_weights_drs'] else None,
                       G_ema if config['ema'] else None)
  
  # which_metrics = ['IS']
  # if not config['no_fid']:
  #   which_metrics += ['FID']
  # if not config['no_intra_fid']:
  #   which_metrics += ['IntraFID']
  # if config['use_lpips']:
  #   which_metrics += ['LPIPS']
  which_metrics = config['which_metrics']
  
  # Burn-In
  print(f'Start BurnIn...')
  D.eval()
  if config['G_eval_mode']:
    G.eval()
    if config['ema']:
      G_ema.eval()
  num_processed_samples = 0
  while num_processed_samples < config['num_burnin_samples']:
    with torch.no_grad():
      z_.sample_()
      y_.sample_()
      if config['parallel']:
        G_z =  nn.parallel.data_parallel(G, (z_, G.shared(y_)))
      else:
        G_z = G(z_, G.shared(y_))
      logits = D(G_z, y_)
      logits = logits.reshape(-1).cpu().numpy()
      batch_ratio = np.exp(logits)
      max_idx = np.argmax(batch_ratio)
      max_ratio = batch_ratio[max_idx]
      if max_ratio > M_state['M']:
        M_state['M'] = max_ratio
        M_state['logit'] = logits[max_idx]
    num_processed_samples += z_.size(0)
  
  # Prepare Sample function for use with inception metrics
  if config['no_drs']:
    sample = functools.partial(utils.sample,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config,
                               use_drs=False)
  else:
    sample = functools.partial(utils.sample_drs,
                               G=(G_ema if config['ema'] and config['use_ema']
                                  else G),
                               z_=z_, y_=y_, config=config,
                               D=D, M_state=M_state,
                               use_deterministic=config['use_deterministic_drs'])
  # Sample
  print('Start Sampling...')
  D.eval()
  if config['G_eval_mode']:
    G.eval()
    if config['ema']:
      G_ema.eval()
  train_fns.test_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                        state_dict, config, experiment_name,
                        config['test_prefix'], config['load_weights_drs'][-1],
                        sample=sample, use_drs=True, drs_classes=drs_classes)
  
  # Test
  print('Start Testing...')
  D.eval()
  if config['G_eval_mode']:
    G.eval()
    if config['ema']:
      G_ema.eval()
  train_fns.test_metric(G, D, G_ema, z_, y_, state_dict, config, sample,
                        get_inception_metrics, experiment_name, test_log,
                        '%s/%s' % (config['logs_root'], experiment_name),
                        config['test_prefix'], config['load_weights_drs'][-1],
                        which_metrics=which_metrics,
                        use_drs=True, drs_classes=drs_classes)


def main():
  # parse command line and run
  parser = utils.prepare_parser()
  config = vars(parser.parse_args())
  # print(config)
  utils.print_config(parser, config)
  run(config)

if __name__ == '__main__':
  main()
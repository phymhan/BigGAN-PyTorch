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
  if config['test_intra_fid_every'] <= 0:
    config['test_intra_fid_every'] = config['test_every']
  if config['test_lpips_every'] <= 0:
    config['test_lpips_every'] = config['test_every']
  # By default, skip init if resuming training.
  if config['resume']:
    print('Skipping initialization for training resumption...')
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
  print(G)
  print(D)
  print('Number of params in G: {} D: {}'.format(
    *[sum([p.data.nelement() for p in net.parameters()]) for net in [G,D]]))
  # Prepare state dict, which holds things like epoch # and itr #
  state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0,
                'best_FID': 999999, 'best_meanFID': 99999, 'best_maxFID': 999999,
                'best_meanLPIPS': -1, 'best_minLPIPS': -1, 'config': config}

  # If loading from a pre-trained model, load weights
  if config['resume']:
    print('Loading weights...')
    utils.load_weights(G, D, state_dict,
                       config['weights_root'], experiment_name, 
                       config['load_weights'][-1] if config['load_weights'] else None,
                       G_ema if config['ema'] else None)

  # If parallel, parallelize the GD module
  if config['parallel']:
    GD = nn.DataParallel(GD)
    if config['cross_replica']:
      patch_replication_callback(GD)

  # Prepare loggers for stats; metrics holds test metrics,
  # lmetrics holds any desired training metrics.
  # test_metrics_fname = '%s/%s_log.jsonl' % (config['logs_root'],
  #                                           experiment_name)
  if config['resume']:
    for fname in ['IS_mean.npy', 'IS_std.npy', 'FID.npy']:
      fname = os.path.join(config['logs_root'], experiment_name, fname)
      if os.path.exists(fname):
        print('{} exists, backing up...'.format(fname))
        shutil.copyfile(fname, fname+'.back')
  else:
    for fname in ['IS_mean.npy', 'IS_std.npy', 'FID.npy']:
      fname = os.path.join(config['logs_root'], experiment_name, fname)
      if os.path.exists(fname):
        print('{} exists, deleting...'.format(fname))
        os.remove(fname)
  test_metrics_fname = '%s/%s_log.txt' % (config['logs_root'], experiment_name)
  train_metrics_fname = '%s/%s' % (config['logs_root'], experiment_name)
  print('Inception Metrics will be saved to {}'.format(test_metrics_fname))
  test_log = utils.MetricsLogger(test_metrics_fname, 
                                 reinitialize=(not config['resume']))
  print('Training Metrics will be saved to {}'.format(train_metrics_fname))
  train_log = utils.MyLogger(train_metrics_fname, 
                             reinitialize=(not config['resume']),
                             logstyle=config['logstyle'])
  if config['use_torch_SN']:
    config['sv_log_interval'] = -1
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
  D_batch_size = (config['batch_size'] * config['num_D_steps']
                  * config['num_D_accumulations'])
  loaders = utils.get_data_loaders(**{**config, 'batch_size': D_batch_size,
                                      'start_itr': state_dict['itr']})

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
  flip_target_hist = utils.str2list(config['fair_label_flip_target_hist']) if config['fair_label_flip_target_hist'] else None
  if flip_target_hist is not None:
    assert(len(flip_target_hist) == config['n_classes'])
    print(f'label flipping target hist: {flip_target_hist}')

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
  # Prepare label statistics
  logpy = utils.prepare_logpy(config['n_classes'], False, config['dataset'], config['data_root'],
                              device=device, fp16=config['D_fp16']) if config['TP'] or config['add_log_ratio_y'] else None
  logqy = utils.prepare_logpy(config['n_classes'], True,  config['dataset'], config['data_root'],
                              device=device, fp16=config['D_fp16']) if config['TQ'] or config['add_log_ratio_y'] else None
  # Loaders are loaded, prepare the training function
  if config['which_train_fn'] == 'GAN':  # original BigGAN training function
    train = train_fns.GAN_training_function(G, D, GD, z_, y_,
                                            ema, state_dict, config)
  elif config['which_train_fn'] == 'fair':
    train = train_fns.fairGAN_training_function(G, D, GD, z_, y_,
                                            ema, state_dict, config)
  elif config['which_train_fn'] == 'hybrid':
    train = train_fns.hybridcGAN_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'scalar':
    train = train_fns.hybridcGAN_adaptive_scalar_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'amortised':
    train = train_fns.hybridcGAN_adaptive_amortised_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'interp_scalar':
    train = train_fns.hybridcGAN_interp_scalar_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'interp_amortised':
    train = train_fns.hybridcGAN_interp_amortised_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'fcGAN':  # where different MI estimators (CE, proj-MINE, eta) are implemented
    train = train_fns.fcGAN_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  elif config['which_train_fn'] == 'naive':
    train = train_fns.naive_hybridcGAN_training_function(G, D, GD, z_, y_,
                                              logpy, logqy,
                                              ema, state_dict, config)
  # Else, assume debugging and use the dummy train fn
  else:
    train = train_fns.dummy_training_function()
  # Prepare Sample function for use with inception metrics
  sample = functools.partial(utils.sample,
                             G=(G_ema if config['ema'] and config['use_ema']
                                else G),
                             z_=z_, y_=y_, config=config)
  
  # Dealing with label flipping
  use_label_flipping = config['fair_use_label_flipping'] and (config['fair_label_flip_rate_range'] and config['fair_label_flip_iter_range'])
  if use_label_flipping:
    flip_rate_delta = ((config['fair_label_flip_rate_range'][1] - config['fair_label_flip_rate_range'][0])/
                       (config['fair_label_flip_iter_range'][1] - config['fair_label_flip_rate_range'][0]))
    flip_rate = config['fair_label_flip_rate_range'][0]
  
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
      G.train()
      D.train()
      if config['ema']:
        G_ema.train()
      if config['D_fp16']:
        x, y = x.to(device).half(), y.to(device)
      else:
        x, y = x.to(device), y.to(device)
      if use_label_flipping:
        if config['fair_label_flip_type'] == 'random':
          y_train = utils.flip_labels(y.clone(), config['n_classes'], flip_rate)
        elif config['fair_label_flip_type'] == 'balance':
          y_train = utils.balance_labels(y.clone(), config['n_classes'], flip_rate, target=flip_target_hist)
        elif config['fair_label_flip_type'] == 'truncate':
          w_train = utils.truncate_labels(y.clone(), config['n_classes'], flip_rate, target=flip_target_hist)
          y_train = (y, w_train)
      else:
        y_train = y
      metrics = train(x, y_train)
      train_log.log(itr=int(state_dict['itr']), **metrics)
      if config['tensorboard']:
        for metric_name in metrics:
          tb_writer.add_scalar('Train/%s' % metric_name, metrics[metric_name], state_dict['itr'])

      # Every sv_log_interval, log singular values
      if (config['sv_log_interval'] > 0) and (not (state_dict['itr'] % config['sv_log_interval'])):
        train_log.log(itr=int(state_dict['itr']), 
                      **{**utils.get_SVs(G, 'G'), **utils.get_SVs(D, 'D')})

      # If using my progbar, print metrics.
      if config['pbar'] == 'mine':
          print(', '.join(['itr: %d' % state_dict['itr']] 
                           + ['%s : %+4.3f' % (key, metrics[key])
                           for key in metrics]), end=' ')

      # Save weights and copies as configured at specified interval
      if not (state_dict['itr'] % config['save_every']):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
          if config['ema']:
            G_ema.eval()
        train_fns.save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                                  state_dict, config, experiment_name)

      # Test every specified interval
      if ((not (state_dict['itr'] % config['test_every'])) or 
          (not config['no_intra_fid'] and not (state_dict['itr'] % config['test_intra_fid_every'])) or 
          (config['use_lpips'] and not (state_dict['itr'] % config['test_lpips_every']))):
        if config['G_eval_mode']:
          print('Switchin G to eval mode...')
          G.eval()
        which_metrics = ['IS']
        if not config['no_fid']:
          which_metrics += ['FID']
        if not config['no_intra_fid']:
          which_metrics += ['IntraFID']
        if config['use_lpips']:
          which_metrics += ['LPIPS']
        train_fns.test(G, D, G_ema, z_, y_, state_dict, config, sample,
                       get_inception_metrics, experiment_name, test_log, tb_writer,
                       '%s/%s' % (config['logs_root'], experiment_name),
                       which_metrics=which_metrics)
      
      # Scheduling label flipping rate
      if use_label_flipping and config['fair_label_flip_rate_range'][0] < state_dict['itr'] < config['fair_label_flip_rate_range'][1]:
        flip_rate += flip_rate_delta
    
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
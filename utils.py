#!/usr/bin/env python
# -*- coding: utf-8 -*-

''' Utilities file
This file contains utility functions for bookkeeping, logging, and data loading.
Methods which directly affect training should either go in layers, the model,
or train_fns.py.
'''

from __future__ import print_function
import sys
import os
import numpy as np
import time
import datetime
import json
import pickle
from argparse import ArgumentParser
import animal_hash
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from layers import identity
import datasets as dset
import shutil
import ast
import pdb

def prepare_parser():
  usage = 'Parser for all scripts.'
  parser = ArgumentParser(description=usage)

  ### Additional ###
  parser.add_argument('--clip_value', type=float, nargs='+', default=[1.], help='clip_value or max_norm, depending on lambda_grad_clip')
  parser.add_argument('--lambda_clip', type=str, default='none', choices=['none', 'value', 'range'])
  parser.add_argument('--lambda_grad_clip', type=str, default='none', choices=['none', 'value', 'norm'])
  parser.add_argument('--gated_lambda', action='store_true')
  parser.add_argument('--no_proj_bias', action='store_true')
  parser.add_argument('--lambda_lr', type=float, default=0.01)
  parser.add_argument('--lambda_penalty_weight', type=float, default=1.)
  parser.add_argument('--max_iters', type=int, default=0)
  parser.add_argument('--mi_weight_decay', type=float, default=1)
  parser.add_argument('--inception_model', type=str, default='default')
  parser.add_argument('--naive_hybrid', action='store_true')
  parser.add_argument('--which_metrics', type=str, nargs='*', default=['IS', 'FID', 'IntraFID', 'LPIPS'], help='Test and DRS')
  parser.add_argument('--index_filename', type=str, default='')
  parser.add_argument('--samples_per_class', type=int, default=10)
  parser.add_argument('--test_G_model_path', type=str, default='')
  parser.add_argument('--num_burnin_samples', type=int, default=10000)
  parser.add_argument('--gamma_percentile', type=float, default=0.60)
  parser.add_argument('--no_drs', action='store_true')
  parser.add_argument('--max_iters_drs', type=int, default=10000)
  parser.add_argument('--use_deterministic_drs', action='store_true')
  parser.add_argument('--fair_drs_classes', type=str, default='', help='which classes to perform DRS?')
  parser.add_argument('--fair_use_label_flipping', action='store_true')
  parser.add_argument('--fair_label_flip_target_hist', type=str, default='', help='target histogram for label flipping')
  parser.add_argument('--fair_label_flip_type', type=str, default='random')
  parser.add_argument('--fair_label_flip_rate_range', type=str2list, default=[])
  parser.add_argument('--fair_label_flip_iter_range', type=str2list, default=[])
  parser.add_argument('--fair_which_div', type=str, default='none', choices=['jsd', 'wgan', 'gan', 'none'])
  parser.add_argument('--fair_weight_div', type=float, default=1e-3)
  parser.add_argument('--use_custom_dataset', action='store_true')
  parser.add_argument('--image_size', type=int, default=64, help='only valid if use_custom_dataset')
  parser.add_argument('--num_classes', type=int, default=2, help='only valid if use_custom_dataset')
  parser.add_argument('--num_classes_per_sheet', type=int, default=50, help='only valid if use_custom_dataset')
  parser.add_argument('--inception_moments_path', type=str, default=None)
  parser.add_argument('--intra_inception_moments_path', type=str, default='')
  parser.add_argument('--use_lpips', action='store_true')
  parser.add_argument('--save_test_iteration', action='store_true')
  parser.add_argument('--test_prefix', type=str, default='test')
  parser.add_argument('--intra_fid_classes', type=str, default='', help='npy file path or list as str')
  parser.add_argument('--lpips_classes', type=str, default='', help='npy file path or list as str')
  parser.add_argument('--torch_fid_num_iters', type=int, default=30)
  parser.add_argument('--load_single_inception_moments', action='store_true', help='load single file for mu and sigma?')
  parser.add_argument('--over_parameterize', action='store_true')
  parser.add_argument('--hybrid_noisy', action='store_true', help='this was a bug')
  parser.add_argument('--detach_weight_linear', action='store_true')
  parser.add_argument('--add_weight_penalty', action='store_true')
  parser.add_argument('--f_div_loss', type=str, default='revkl')
  parser.add_argument('--use_torch_SN', action='store_true')
  parser.add_argument('--custom_inception_model_path', type=str, default='', help='if not empty, load custom inception model')
  parser.add_argument('--custom_num_classes', type=int, default=1000, help='num classes')
  parser.add_argument('--optimizer', type=str, default='adam', help='([sgd | adam] default: %(default)s)')
  # parser.add_argument('--debug_mode', action='store_true')
  parser.add_argument('--use_hybrid', action='store_true')
  parser.add_argument('--use_scaled_bce_logits_with_weighted_ce_loss', action='store_true')
  parser.add_argument('--adaptive_loss', type=str, default='', help='([none | sigma | softmax])')
  parser.add_argument('--adaptive_loss_param', type=str, default='none', help='([none | sn])')
  parser.add_argument('--adaptive_loss_detach', action='store_true', help='detach weight when training G?')
  parser.add_argument('--adaptive_gen_loss', action='store_true')  # TODO: will be removed in clean version
  parser.add_argument('--adaptive_gen_loss_detach', action='store_true')  # TODO: will be removed in clean version
  parser.add_argument('--adaptive_gen_loss_ignore_log_penalty', action='store_true')  # TODO: will be removed in clean version
  parser.add_argument('--weighted_hinge_loss', type=str, default='inside')  # TODO: will be removed in clean version
  parser.add_argument('--MI_P_type', type=str, default='ce', help='([ce | mine | eta] default: %(default)s)')
  parser.add_argument('--MI_Q_type', type=str, default='ce', help='([ce | mine | eta] default: %(default)s)')
  parser.add_argument('--no_add_bias', action='store_false', dest='add_bias', default=True,
                      help='Add bias in TP and TQ? (default: %(default)s)')
  parser.add_argument('--linear_no_sn', action='store_true')
  parser.add_argument('--add_log_ratio_y', action='store_true')
  parser.add_argument('--psi_dim', type=int, nargs='*', default=[1], help='fc dims for linear layers in T')
  parser.add_argument('--dis_fc_dim', type=int, nargs='*', default=[1], help='fc dims for dis_fc')
  parser.add_argument('--log_sigma_dim', type=int, nargs='*', default=[1], help='fc dims for log sigma')
  parser.add_argument('--log_sigma_sn', action='store_false', dest='log_sigma_no_sn', default=True)
  parser.add_argument('--log_sigma_nl', type=str, default='softplus',
    help='Add non-linear after last linear [relu | none | tanh | softplus]')
  parser.add_argument('--use_torch', action='store_true', dest='use_torch', default=False,
                      help='Use torch when computing FID? (default: %(default)s)')
  parser.add_argument('--use_torch_intra', action='store_true', dest='use_torch_intra', default=False,
                      help='Use torch when computing Intra-FID? (default: %(default)s)')
  parser.add_argument(
    '--tensorboard', action='store_true', default=False,
    help='Use Tensorboard? (default: %(default)s)')
  parser.add_argument(
    '--GAN_loss', type=str, default='hinge',
    help='GAN loss ([hinge | dcgan | vanilla | lsgan] default: %(default)s)')
  parser.add_argument(
    '--MI_loss', type=str, default='identity',
    help='MI loss ([hinge | identity] default: %(default)s)')
  parser.add_argument(
    '--loss_type', type=str, default='Projection',
    help='Loss type ([Projection | AC | TAC | MINE | fCGAN | WDM] default: %(default)s)')
  parser.add_argument(
    '--no_projection', action='store_false', dest='projection', default=True,
    help='Use Projection? (default: %(default)s)')
  parser.add_argument(
    '--AC', action='store_true', default=False,
    help='Use AC? (default: %(default)s)')
  parser.add_argument(
    '--TAC', action='store_true', default=False,
    help='Use Twin AC? (default: %(default)s)')
  parser.add_argument(
    '--TP', action='store_true', default=False,
    help='Use MINE for real data? (default: %(default)s)')
  parser.add_argument(
    '--TQ', action='store_true', default=False,
    help='Use MINE for fake data? (default: %(default)s)')
  parser.add_argument(
    '--use_softmax', action='store_true', default=False,
    help='Use softmax in MINE? (default: %(default)s)')
  parser.add_argument(
    '--AC_weight', type=float, default=1.0,
    help='Weight for AC loss (default: %(default)s)')
  parser.add_argument(
    '--MI_weight', type=float, default=1.0,
    help='Weight for MINE or fCGAN loss (default: %(default)s)')
  parser.add_argument(
    '--train_AC_on_fake', action='store_true', default=False,
    help='Train AC on fake? (default: %(default)s)')
  
  ### Dataset/Dataloader stuff ###
  parser.add_argument(
    '--dataset', type=str, default='I128_hdf5',
    help='Which Dataset to train on, out of I128, I256, C10, C100;'
         'Append "_hdf5" to use the hdf5 version for ISLVRC '
         '(default: %(default)s)')
  parser.add_argument(
    '--augment', action='store_true', default=False,
    help='Augment with random crops and flips (default: %(default)s)')
  parser.add_argument(
    '--num_workers', type=int, default=8,
    help='Number of dataloader workers; consider using less for HDF5 '
         '(default: %(default)s)')
  parser.add_argument(
    '--no_pin_memory', action='store_false', dest='pin_memory', default=True,
    help='Pin data into memory through dataloader? (default: %(default)s)')
  parser.add_argument(
    '--shuffle', action='store_true', default=False,
    help='Shuffle the data (strongly recommended)? (default: %(default)s)')
  parser.add_argument(
    '--load_in_mem', action='store_true', default=False,
    help='Load all data into memory? (default: %(default)s)')
  parser.add_argument(
    '--use_multiepoch_sampler', action='store_true', default=False,
    help='Use the multi-epoch sampler for dataloader? (default: %(default)s)')

  ### Model stuff ###
  parser.add_argument(
    '--model', type=str, default='BigGAN',
    help='Name of the model module (default: %(default)s)')
  parser.add_argument(
    '--G_param', type=str, default='SN',
    help='Parameterization style to use for G, spectral norm (SN) or SVD (SVD)'
          ' or None (default: %(default)s)')
  parser.add_argument(
    '--D_param', type=str, default='SN',
    help='Parameterization style to use for D, spectral norm (SN) or SVD (SVD)'
         ' or None (default: %(default)s)')    
  parser.add_argument(
    '--G_ch', type=int, default=64,
    help='Channel multiplier for G (default: %(default)s)')
  parser.add_argument(
    '--D_ch', type=int, default=64,
    help='Channel multiplier for D (default: %(default)s)')
  parser.add_argument(
    '--G_depth', type=int, default=1,
    help='Number of resblocks per stage in G? (default: %(default)s)')
  parser.add_argument(
    '--D_depth', type=int, default=1,
    help='Number of resblocks per stage in D? (default: %(default)s)')
  parser.add_argument(
    '--D_thin', action='store_false', dest='D_wide', default=True,
    help='Use the SN-GAN channel pattern for D? (default: %(default)s)')
  parser.add_argument(
    '--G_shared', action='store_true', default=False,
    help='Use shared embeddings in G? (default: %(default)s)')
  parser.add_argument(
    '--shared_dim', type=int, default=0,
    help='G''s shared embedding dimensionality; if 0, will be equal to dim_z. '
         '(default: %(default)s)')
  parser.add_argument(
    '--dim_z', type=int, default=128,
    help='Noise dimensionality: %(default)s)')
  parser.add_argument(
    '--z_var', type=float, default=1.0,
    help='Noise variance: %(default)s)')    
  parser.add_argument(
    '--hier', action='store_true', default=False,
    help='Use hierarchical z in G? (default: %(default)s)')
  parser.add_argument(
    '--cross_replica', action='store_true', default=False,
    help='Cross_replica batchnorm in G?(default: %(default)s)')
  parser.add_argument(
    '--mybn', action='store_true', default=False,
    help='Use my batchnorm (which supports standing stats?) %(default)s)')
  parser.add_argument(
    '--G_nl', type=str, default='relu',
    help='Activation function for G (default: %(default)s)')
  parser.add_argument(
    '--D_nl', type=str, default='relu',
    help='Activation function for D (default: %(default)s)')
  parser.add_argument(
    '--G_attn', type=str, default='64',
    help='What resolutions to use attention on for G (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--D_attn', type=str, default='64',
    help='What resolutions to use attention on for D (underscore separated) '
         '(default: %(default)s)')
  parser.add_argument(
    '--norm_style', type=str, default='bn',
    help='Normalizer style for G, one of bn [batchnorm], in [instancenorm], '
         'ln [layernorm], gn [groupnorm] (default: %(default)s)')
         
  ### Model init stuff ###
  parser.add_argument(
    '--seed', type=int, default=0,
    help='Random seed to use; affects both initialization and '
         ' dataloading. (default: %(default)s)')
  parser.add_argument(
    '--G_init', type=str, default='ortho',
    help='Init style to use for G (default: %(default)s)')
  parser.add_argument(
    '--D_init', type=str, default='ortho',
    help='Init style to use for D(default: %(default)s)')
  parser.add_argument(
    '--skip_init', action='store_true', default=False,
    help='Skip initialization, ideal for testing when ortho init was used '
          '(default: %(default)s)')
  
  ### Optimizer stuff ###
  parser.add_argument(
    '--G_lr', type=float, default=5e-5,
    help='Learning rate to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_lr', type=float, default=2e-4,
    help='Learning rate to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B1', type=float, default=0.0,
    help='Beta1 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B1', type=float, default=0.0,
    help='Beta1 to use for Discriminator (default: %(default)s)')
  parser.add_argument(
    '--G_B2', type=float, default=0.999,
    help='Beta2 to use for Generator (default: %(default)s)')
  parser.add_argument(
    '--D_B2', type=float, default=0.999,
    help='Beta2 to use for Discriminator (default: %(default)s)')
    
  ### Batch size, parallel, and precision stuff ###
  parser.add_argument(
    '--batch_size', type=int, default=64,
    help='Default overall batchsize (default: %(default)s)')
  parser.add_argument(
    '--G_batch_size', type=int, default=0,
    help='Batch size to use for G; if 0, same as D (default: %(default)s)')
  parser.add_argument(
    '--num_G_accumulations', type=int, default=1,
    help='Number of passes to accumulate G''s gradients over '
         '(default: %(default)s)')  
  parser.add_argument(
    '--num_D_steps', type=int, default=2,
    help='Number of D steps per G step (default: %(default)s)')
  parser.add_argument(
    '--num_D_accumulations', type=int, default=1,
    help='Number of passes to accumulate D''s gradients over '
         '(default: %(default)s)')
  parser.add_argument(
    '--split_D', action='store_true', default=False,
    help='Run D twice rather than concatenating inputs? (default: %(default)s)')
  parser.add_argument(
    '--num_epochs', type=int, default=100,
    help='Number of epochs to train for (default: %(default)s)')
  parser.add_argument(
    '--parallel', action='store_true', default=False,
    help='Train with multiple GPUs (default: %(default)s)')
  parser.add_argument(
    '--G_fp16', action='store_true', default=False,
    help='Train with half-precision in G? (default: %(default)s)')
  parser.add_argument(
    '--D_fp16', action='store_true', default=False,
    help='Train with half-precision in D? (default: %(default)s)')
  parser.add_argument(
    '--D_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--G_mixed_precision', action='store_true', default=False,
    help='Train with half-precision activations but fp32 params in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--accumulate_stats', action='store_true', default=False,
    help='Accumulate "standing" batchnorm stats? (default: %(default)s)')
  parser.add_argument(
    '--num_standing_accumulations', type=int, default=16,
    help='Number of forward passes to use in accumulating standing stats? '
         '(default: %(default)s)')        
    
  ### Bookkeping stuff ###  
  parser.add_argument(
    '--G_eval_mode', action='store_true', default=False,
    help='Run G in eval mode (running/standing stats?) at sample/test time? '
         '(default: %(default)s)')
  parser.add_argument(
    '--save_every', type=int, default=2000,
    help='Save every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_save_copies', type=int, default=2,
    help='How many copies to save (default: %(default)s)')
  parser.add_argument(
    '--num_best_copies', type=int, default=2,
    help='How many previous best checkpoints to save (default: %(default)s)')
  parser.add_argument(
    '--which_best', type=str, default='IS',
    help='Which metric to use to determine when to save new "best"'
         'checkpoints, one of IS or FID (default: %(default)s)')
  parser.add_argument(
    '--no_fid', action='store_true', default=False,
    help='Calculate IS only, not FID? (default: %(default)s)')
  parser.add_argument(
    '--no_intra_fid', action='store_true', default=False,
    help='Do not calculate Intra-FID? (default: %(default)s)')
  parser.add_argument(
    '--test_every', type=int, default=5000,
    help='Test every X iterations (default: %(default)s)')
  parser.add_argument(
    '--test_intra_fid_every', type=int, default=0,
    help='Test intra-FID every X iterations (default: %(default)s)')
  parser.add_argument(
    '--test_lpips_every', type=int, default=0,
    help='Test LPIPS every X iterations (default: %(default)s)')
  parser.add_argument(
    '--num_inception_images', type=int, default=50000,
    help='Number of samples to compute inception metrics with '
         '(default: %(default)s)')
  parser.add_argument('--num_intra_inception_images', type=int, default=2000,
    help='Number of samples to compute inception metrics with (default: %(default)s)')
  parser.add_argument('--num_lpips_images', type=int, default=50000,
    help='Number of samples to compute inception metrics with (default: %(default)s)')
  parser.add_argument(
    '--hashname', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)') 
  parser.add_argument(
    '--base_root', type=str, default='',
    help='Default location to store all weights, samples, data, and logs '
           ' (default: %(default)s)')
  parser.add_argument(
    '--data_root', type=str, default='data',
    help='Default location where data is stored (default: %(default)s)')
  parser.add_argument(
    '--weights_root', type=str, default='weights',
    help='Default location to store weights (default: %(default)s)')
  parser.add_argument(
    '--logs_root', type=str, default='logs',
    help='Default location to store logs (default: %(default)s)')
  parser.add_argument(
    '--samples_root', type=str, default='samples',
    help='Default location to store samples (default: %(default)s)')  
  parser.add_argument(
    '--pbar', type=str, default='mine',
    help='Type of progressbar to use; one of "mine" or "tqdm" '
         '(default: %(default)s)')
  parser.add_argument(
    '--name_suffix', type=str, default='',
    help='Suffix for experiment name for loading weights for sampling '
         '(consider "best0") (default: %(default)s)')
  parser.add_argument(
    '--experiment_name', type=str, default='',
    help='Optionally override the automatic experiment naming with this arg. '
         '(default: %(default)s)')
  parser.add_argument(
    '--config_from_name', action='store_true', default=False,
    help='Use a hash of the experiment name instead of the full config '
         '(default: %(default)s)')
         
  ### EMA Stuff ###
  parser.add_argument(
    '--ema', action='store_true', default=False,
    help='Keep an ema of G''s weights? (default: %(default)s)')
  parser.add_argument(
    '--ema_decay', type=float, default=0.9999,
    help='EMA decay rate (default: %(default)s)')
  parser.add_argument(
    '--use_ema', action='store_true', default=False,
    help='Use the EMA parameters of G for evaluation? (default: %(default)s)')
  parser.add_argument(
    '--ema_start', type=int, default=0,
    help='When to start updating the EMA weights (default: %(default)s)')
  
  ### Numerical precision and SV stuff ### 
  parser.add_argument(
    '--adam_eps', type=float, default=1e-8,
    help='epsilon value to use for Adam (default: %(default)s)')
  parser.add_argument(
    '--BN_eps', type=float, default=1e-5,
    help='epsilon value to use for BatchNorm (default: %(default)s)')
  parser.add_argument(
    '--SN_eps', type=float, default=1e-8,
    help='epsilon value to use for Spectral Norm(default: %(default)s)')
  parser.add_argument(
    '--num_G_SVs', type=int, default=1,
    help='Number of SVs to track in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SVs', type=int, default=1,
    help='Number of SVs to track in D (default: %(default)s)')
  parser.add_argument(
    '--num_G_SV_itrs', type=int, default=1,
    help='Number of SV itrs in G (default: %(default)s)')
  parser.add_argument(
    '--num_D_SV_itrs', type=int, default=1,
    help='Number of SV itrs in D (default: %(default)s)')
  
  ### Ortho reg stuff ### 
  parser.add_argument(
    '--G_ortho', type=float, default=0.0, # 1e-4 is default for BigGAN
    help='Modified ortho reg coefficient in G(default: %(default)s)')
  parser.add_argument(
    '--D_ortho', type=float, default=0.0,
    help='Modified ortho reg coefficient in D (default: %(default)s)')
  parser.add_argument(
    '--toggle_grads', action='store_true', default=True,
    help='Toggle D and G''s "requires_grad" settings when not training them? '
         ' (default: %(default)s)')
  
  ### Which train function ###
  parser.add_argument(
    '--which_train_fn', type=str, default='ACGAN',
    help='How2trainyourbois (default: %(default)s)')  
  
  ### Resume training stuff
  parser.add_argument(
    '--load_weights', type=str, default=[], nargs='*',
    help='Suffix for which weights to load (e.g. best0, copy0) '
         '(default: %(default)s)')
  parser.add_argument(
    '--load_weights_drs', type=str, default=['drs_itr1000'], nargs='*',
    help='Suffix for which weights to load (e.g. drs_itr1000) '
         '(default: %(default)s)')
  # parser.add_argument(
  #   '--load_weights', type=str, default='',
  #   help='Suffix for which weights to load (e.g. best0, copy0) '
  #        '(default: %(default)s)')
  parser.add_argument(
    '--resume', action='store_true', default=False,
    help='Resume training? (default: %(default)s)')
  
  ### Log stuff ###
  parser.add_argument(
    '--logstyle', type=str, default='%3.3e',
    help='What style to use when logging training metrics?'
         'One of: %#.#f/ %#.#e (float/exp, text),'
         'pickle (python pickle),'
         'npz (numpy zip),'
         'mat (MATLAB .mat file) (default: %(default)s)')
  parser.add_argument(
    '--log_G_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in G? '
         '(default: %(default)s)')
  parser.add_argument(
    '--log_D_spectra', action='store_true', default=False,
    help='Log the top 3 singular values in each SN layer in D? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sv_log_interval', type=int, default=10,
    help='Iteration interval for logging singular values '
         ' (default: %(default)s)')
  return parser

# Arguments for sample.py; not presently used in train.py
def add_sample_parser(parser):
  parser.add_argument(
    '--sample_npz', action='store_true', default=False,
    help='Sample "sample_num_npz" images and save to npz? '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_num_npz', type=int, default=50000,
    help='Number of images to sample when sampling NPZs '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_sheets', action='store_true', default=False,
    help='Produce class-conditional sample sheets and stick them in '
         'the samples root? (default: %(default)s)')
  parser.add_argument(
    '--sample_interps', action='store_true', default=False,
    help='Produce interpolation sheets and stick them in '
         'the samples root? (default: %(default)s)')         
  parser.add_argument(
    '--sample_sheet_folder_num', type=int, default=-1,
    help='Number to use for the folder for these sample sheets '
         '(default: %(default)s)')
  parser.add_argument(
    '--sample_random', action='store_true', default=False,
    help='Produce a single random sheet? (default: %(default)s)')
  parser.add_argument(
    '--sample_trunc_curves', type=str, default='',
    help='Get inception metrics with a range of variances?'
         'To use this, specify a startpoint, step, and endpoint, e.g. '
         '--sample_trunc_curves 0.2_0.1_1.0 for a startpoint of 0.2, '
         'endpoint of 1.0, and stepsize of 1.0.  Note that this is '
         'not exactly identical to using tf.truncated_normal, but should '
         'have approximately the same effect. (default: %(default)s)')
  parser.add_argument(
    '--sample_inception_metrics', action='store_true', default=False,
    help='Calculate Inception metrics with sample.py? (default: %(default)s)')  
  return parser

# Convenience dicts
dset_dict = {'I32': dset.ImageFolder, 'I64': dset.ImageFolder,
             'I128': dset.ImageFolder, 'I256': dset.ImageFolder,
             'I32_hdf5': dset.ILSVRC_HDF5, 'I64_hdf5': dset.ILSVRC_HDF5,
             'I128_hdf5': dset.ILSVRC_HDF5, 'I256_hdf5': dset.ILSVRC_HDF5,
             'T64': dset.ImageFolder,
             'M64': dset.ImageFolder, 'M128': dset.ImageFolder,
             'C10': dset.CIFAR10, 'C100': dset.CIFAR100, 'C100IB': dset.ImageFolder,
             'V200': dset.ImageFolder, 'V200_hdf5': dset.ILSVRC_HDF5,
             'V500': dset.ImageFolder, 'V500_hdf5': dset.ILSVRC_HDF5,
             'V1000': dset.ImageFolder, 'V1000_hdf5': dset.ILSVRC_HDF5,
             'V2000': dset.ImageFolder, 'V2000_hdf5': dset.ILSVRC_HDF5}
imsize_dict = {'I32': 32, 'I32_hdf5': 32,
               'I64': 64, 'I64_hdf5': 64,
               'I128': 128, 'I128_hdf5': 128,
               'I256': 256, 'I256_hdf5': 256,
               'T64': 64, 'M64': 64, 'M128': 128,
               'C10': 32, 'C100': 32, 'C100IB': 32,
               'V200': 64, 'V200_hdf5': 64,
               'V500': 64, 'V500_hdf5': 64,
               'V1000': 64, 'V1000_hdf5': 64,
               'V2000': 64, 'V2000_hdf5': 64}
root_dict = {'I32': 'ImageNet', 'I32_hdf5': 'ILSVRC32.hdf5',
             'I64': 'ImageNet', 'I64_hdf5': 'ILSVRC64.hdf5',
             'I128': 'ImageNet', 'I128_hdf5': 'ILSVRC128.hdf5',
             'I256': 'ImageNet', 'I256_hdf5': 'ILSVRC256.hdf5',
             'T64': 'Tiny-ImageNet-200',
             'M64': 'Mini-ImageNet-100', 'M128': 'Mini-ImageNet-100',
             'C10': 'cifar', 'C100': 'cifar', 'C100IB': 'cifar100-imbalanced',
             'V200': 'V200', 'V200_hdf5': 'VGGFace200_64.hdf5',
             'V500': 'V500', 'V500_hdf5': 'VGGFace500_64.hdf5',
             'V1000': 'V1000', 'V1000_hdf5': 'VGGFace1000_64.hdf5',
             'V2000': 'V2000', 'V2000_hdf5': 'VGGFace2000_64.hdf5'}
nclass_dict = {'I32': 1000, 'I32_hdf5': 1000,
               'I64': 1000, 'I64_hdf5': 1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'T64': 200,
               'M64': 100, 'M128': 100,
               'C10': 10, 'C100': 100, 'C100IB': 100,
               'V200': 200, 'V200_hdf5': 200,
               'V500': 500, 'V500_hdf5': 500,
               'V1000': 1000, 'V1000_hdf5': 1000,
               'V2000': 2000, 'V2000_hdf5': 2000}
# Number of classes to put per sample sheet               
classes_per_sheet_dict = {'I32': 50, 'I32_hdf5': 50,
                          'I64': 50, 'I64_hdf5': 50,
                          'I128': 20, 'I128_hdf5': 20,
                          'I256': 20, 'I256_hdf5': 20,
                          'T64': 50, 'M64': 50, 'M128': 50,
                          'C10': 10, 'C100': 100, 'C100IB': 100,
                          'V200': 100, 'V200_hdf5': 100,
                          'V500': 100, 'V500_hdf5': 100,
                          'V1000': 100, 'V1000_hdf5': 100,
                          'V2000': 100, 'V2000_hdf5': 100}
activation_dict = {'inplace_relu': nn.ReLU(inplace=True),
                   'relu': nn.ReLU(inplace=False),
                   'ir': nn.ReLU(inplace=True),
                   'none': identity(),
                   'softplus': nn.Softplus()}

class CenterCropLongEdge(object):
  """Crops the given PIL Image on the long edge.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    return transforms.functional.center_crop(img, min(img.size))

  def __repr__(self):
    return self.__class__.__name__

class RandomCropLongEdge(object):
  """Crops the given PIL Image on the long edge with a random start point.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
  """
  def __call__(self, img):
    """
    Args:
        img (PIL Image): Image to be cropped.
    Returns:
        PIL Image: Cropped image.
    """
    size = (min(img.size), min(img.size))
    # Only step forward along this edge if it's the long edge
    i = (0 if size[0] == img.size[0] 
          else np.random.randint(low=0,high=img.size[0] - size[0]))
    j = (0 if size[1] == img.size[1]
          else np.random.randint(low=0,high=img.size[1] - size[1]))
    return transforms.functional.crop(img, j, i, size[0], size[1])

  def __repr__(self):
    return self.__class__.__name__

    
# multi-epoch Dataset sampler to avoid memory leakage and enable resumption of
# training from the same sample regardless of if we stop mid-epoch
class MultiEpochSampler(torch.utils.data.Sampler):
  r"""Samples elements randomly over multiple epochs

  Arguments:
      data_source (Dataset): dataset to sample from
      num_epochs (int) : Number of times to loop over the dataset
      start_itr (int) : which iteration to begin from
  """

  def __init__(self, data_source, num_epochs, start_itr=0, batch_size=128):
    self.data_source = data_source
    self.num_samples = len(self.data_source)
    self.num_epochs = num_epochs
    self.start_itr = start_itr
    self.batch_size = batch_size

    if not isinstance(self.num_samples, int) or self.num_samples <= 0:
      raise ValueError("num_samples should be a positive integeral "
                       "value, but got num_samples={}".format(self.num_samples))

  def __iter__(self):
    n = len(self.data_source)
    # Determine number of epochs
    num_epochs = int(np.ceil((n * self.num_epochs 
                              - (self.start_itr * self.batch_size)) / float(n)))
    # Sample all the indices, and then grab the last num_epochs index sets;
    # This ensures if we're starting at epoch 4, we're still grabbing epoch 4's
    # indices
    out = [torch.randperm(n) for epoch in range(self.num_epochs)][-num_epochs:]
    # Ignore the first start_itr % n indices of the first epoch
    out[0] = out[0][(self.start_itr * self.batch_size % n):]
    # if self.replacement:
      # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
    # return iter(.tolist())
    output = torch.cat(out).tolist()
    print('Length dataset output is %d' % len(output))
    return iter(output)

  def __len__(self):
    return len(self.data_source) * self.num_epochs - self.start_itr * self.batch_size


# Convenience function to centralize all data loaders
def get_data_loaders(dataset, data_root=None, augment=False, batch_size=64, 
                     num_workers=8, shuffle=True, load_in_mem=False, hdf5=False,
                     pin_memory=True, drop_last=True, start_itr=0,
                     num_epochs=500, use_multiepoch_sampler=False,
                     use_custom_dataset=False, image_size=0, index_filename='',
                     return_dataset=False,
                     **kwargs):

  # Append /FILENAME.hdf5 to root if using hdf5
  if not use_custom_dataset:
    data_root += '/%s' % root_dict[dataset]
    print('Using dataset root location %s' % data_root)
    which_dataset = dset_dict[dataset]
    image_size = imsize_dict[dataset]
  else:
    if data_root.endswith('.hdf5'):
      which_dataset = dset.ILSVRC_HDF5
    else:
      which_dataset = dset.ImageFolder
    image_size = image_size
  norm_mean = [0.5,0.5,0.5]
  norm_std = [0.5,0.5,0.5]
  
  # For image folder datasets, name of the file where we store the precomputed
  # image locations to avoid having to walk the dirs every time we load.
  dataset_kwargs = {'index_filename': index_filename or '%s_imgs.npz' % dataset}
  
  # HDF5 datasets have their own inbuilt transform, no need to train_transform  
  if ('hdf5' in dataset) or data_root.endswith('.hdf5'):
    train_transform = None
  else:
    if augment:
      print('Data will be augmented...')
      if dataset in ['C10', 'C100', 'C100IB']:
        train_transform = [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()]
      else:
        train_transform = [RandomCropLongEdge(),
                           transforms.Resize(image_size),
                           transforms.RandomHorizontalFlip()]
    else:
      print('Data will not be augmented...')
      if dataset in ['C10', 'C100', 'C100IB']:
        train_transform = []
      else:
        train_transform = [CenterCropLongEdge(), transforms.Resize(image_size)]
      # train_transform = [transforms.Resize(image_size), transforms.CenterCrop]
    train_transform = transforms.Compose(train_transform + [
                      transforms.ToTensor(),
                      transforms.Normalize(norm_mean, norm_std)])
  train_set = which_dataset(root=data_root, transform=train_transform,
                            load_in_mem=load_in_mem, **dataset_kwargs)
  if return_dataset:
    return train_set
  # Prepare loader; the loaders list is for forward compatibility with
  # using validation / test splits.
  loaders = []
  drop_last = True
  if use_multiepoch_sampler:
    print('Using multiepoch sampler from start_itr %d...' % start_itr)
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
      'drop_last': drop_last}
    sampler = MultiEpochSampler(train_set, num_epochs, start_itr, batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=sampler, **loader_kwargs)
  else:
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory,
                     'drop_last': drop_last} # Default, drop last incomplete batch
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, **loader_kwargs)
  loaders.append(train_loader)
  return loaders


# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


# Utility to peg all roots to a base root
# If a base root folder is provided, peg all other root folders to it.
def update_config_roots(config):
  if config['base_root']:
    print('Pegging all root folders to base root %s' % config['base_root'])
    for key in ['data', 'weights', 'logs', 'samples']:
      config['%s_root' % key] = '%s/%s' % (config['base_root'], key)
  return config


# Utility to prepare root folders if they don't exist; parent folder must exist
def prepare_root(config):
  for key in ['weights_root', 'logs_root', 'samples_root']:
    if not os.path.exists(config[key]):
      print('Making directory %s for %s...' % (config[key], key))
      os.makedirs(config[key])


# Simple wrapper that applies EMA to a model. COuld be better done in 1.0 using
# the parameters() and buffers() module functions, but for now this works
# with state_dicts using .copy_
class ema(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay 
                                     + self.source_dict[key].data * (1 - decay))


# Apply modified ortho reg to a model
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)


# Default ortho reg
# This function is an optimized version that directly computes the gradient,
# instead of computing and then differentiating the loss.
def default_ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes & not in blacklist
      if len(param.shape) < 2 or param in blacklist:
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t()) 
               - torch.eye(w.shape[0], device=w.device), w))
      param.grad.data += strength * grad.view(param.shape)


# Convenience utility to switch off requires_grad
def toggle_grad(model, on_or_off):
  for param in model.parameters():
    param.requires_grad = on_or_off


# Function to join strings or ignore them
# Base string is the string to link "strings," while strings
# is a list of strings or Nones.
def join_strings(base_string, strings):
  return base_string.join([item for item in strings if item])


# Save a model's weights, optimizer, and the state_dict
def save_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None):
  root = '/'.join([weights_root, experiment_name])
  if not os.path.exists(root):
    os.mkdir(root)
  if name_suffix:
    print('Saving weights to %s/%s...' % (root, name_suffix))
  else:
    print('Saving weights to %s...' % root)
  if G is not None:
    torch.save(G.state_dict(), 
                '%s/%s.pth' % (root, join_strings('_', ['G', name_suffix])))
    torch.save(G.optim.state_dict(), 
                '%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix])))
  if D is not None:
    torch.save(D.state_dict(), 
                '%s/%s.pth' % (root, join_strings('_', ['D', name_suffix])))
    torch.save(D.optim.state_dict(),
                '%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix])))
  if state_dict is not None:
    torch.save(state_dict,
                '%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))
  if G_ema is not None:
    torch.save(G_ema.state_dict(), 
                '%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix])))


# Load a model's weights, optimizer, and the state_dict
def load_weights(G, D, state_dict, weights_root, experiment_name, 
                 name_suffix=None, G_ema=None, strict=True, load_optim=True, model_path=None):
  root = '/'.join([weights_root, experiment_name])
  if name_suffix:
    print('Loading %s weights from %s...' % (name_suffix, root))
  else:
    print('Loading weights from %s...' % root)
  if G is not None:
    G.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G', name_suffix]))),
      strict=strict)
    if load_optim:
      G.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['G_optim', name_suffix]))))
  if D is not None:
    D.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['D', name_suffix]))),
      strict=strict)
    if load_optim:
      D.optim.load_state_dict(
        torch.load('%s/%s.pth' % (root, join_strings('_', ['D_optim', name_suffix]))))
  # Load state dict
  if state_dict is not None:
    state_dict_loaded = torch.load('%s/%s.pth' % (root, join_strings('_', ['state_dict', name_suffix])))
    for item in state_dict:
      if item in state_dict_loaded:
        state_dict[item] = state_dict_loaded[item]
  if G_ema is not None:
    G_ema.load_state_dict(
      torch.load('%s/%s.pth' % (root, join_strings('_', ['G_ema', name_suffix]))), strict=strict)


''' MetricsLogger originally stolen from VoxNet source code.
    Used for logging inception metrics'''
class MetricsLogger(object):
  def __init__(self, fname, reinitialize=False):
    self.fname = fname
    self.reinitialize = reinitialize
    if os.path.exists(self.fname):
      if self.reinitialize:
        print('{} exists, deleting...'.format(self.fname))
        os.remove(self.fname)

  def log(self, record=None, **kwargs):
    """
    Assumption: no newlines in the input.
    """
    if record is None:
      record = {}
    record.update(kwargs)
    record['_stamp'] = time.time()
    with open(self.fname, 'a') as f:
      f.write(json.dumps(record, ensure_ascii=True) + '\n')


# Logstyle is either:
# '%#.#f' for floating point representation in text
# '%#.#e' for exponent representation in text
# 'npz' for output to npz # NOT YET SUPPORTED
# 'pickle' for output to a python pickle # NOT YET SUPPORTED
# 'mat' for output to a MATLAB .mat file # NOT YET SUPPORTED
class MyLogger(object):
  def __init__(self, fname, reinitialize=False, logstyle='%3.3f'):
    self.root = fname
    if not os.path.exists(self.root):
      os.mkdir(self.root)
    self.reinitialize = reinitialize
    self.metrics = []
    self.logstyle = logstyle # One of '%3.3f' or like '%3.3e'

  # Delete log if re-starting and log already exists
  def reinit(self, item):
    if os.path.exists('%s/%s.log' % (self.root, item)):
      if self.reinitialize:
        # Only print the removal mess
        if 'sv' in item :
          if not any('sv' in item for item in self.metrics):
            print('Deleting singular value logs...')
        else:
          print('{} exists, deleting...'.format('%s_%s.log' % (self.root, item)))
        os.remove('%s/%s.log' % (self.root, item))
  
  # Log in plaintext; this is designed for being read in MATLAB(sorry not sorry)
  def log(self, itr, **kwargs):
    for arg in kwargs:
      if arg not in self.metrics:
        if self.reinitialize:
          self.reinit(arg)
        self.metrics += [arg]
      if self.logstyle == 'pickle':
        print('Pickle not currently supported...')
         # with open('%s/%s.log' % (self.root, arg), 'a') as f:
          # pickle.dump(kwargs[arg], f)
      elif self.logstyle == 'mat':
        print('.mat logstyle not currently supported...')
      else:
        with open('%s/%s.log' % (self.root, arg), 'a') as f:
          f.write('%d: %s\n' % (itr, self.logstyle % kwargs[arg]))


# Write some metadata to the logs directory
def write_metadata(logs_root, experiment_name, config, state_dict):
  with open(('%s/%s/metalog.txt' % 
             (logs_root, experiment_name)), 'w') as writefile:
    writefile.write('datetime: %s\n' % str(datetime.datetime.now()))
    writefile.write('config: %s\n' % str(config))
    writefile.write('state: %s\n' %str(state_dict))


"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
Andy's adds: time elapsed in addition to ETA, makes it possible to add
estimated time to 1k iters instead of estimated time to completion.
"""
def progress(items, desc='', total=None, min_delay=0.1, displaytype='s1k'):
  """
  Returns a generator over `items`, printing the number and percentage of
  items processed and the estimated remaining processing time before yielding
  the next item. `total` gives the total number of items (required if `items`
  has no length), and `min_delay` gives the minimum time in seconds between
  subsequent prints. `desc` gives an optional prefix text (end with a space).
  """
  total = total or len(items)
  t_start = time.time()
  t_last = 0
  for n, item in enumerate(items):
    t_now = time.time()
    if t_now - t_last > min_delay:
      print("\r%s%d/%d (%6.2f%%)" % (
              desc, n+1, total, n / float(total) * 100), end=" ")
      if n > 0:
        
        if displaytype == 's1k': # minutes/seconds for 1000 iters
          next_1000 = n + (1000 - n%1000)
          t_done = t_now - t_start
          t_1k = t_done / n * next_1000
          outlist = list(divmod(t_done, 60)) + list(divmod(t_1k - t_done, 60))
          print("(TE/ET1k: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
        else:# displaytype == 'eta':
          t_done = t_now - t_start
          t_total = t_done / n * total
          outlist = list(divmod(t_done, 60)) + list(divmod(t_total - t_done, 60))
          print("(TE/ETA: %d:%02d / %d:%02d)" % tuple(outlist), end=" ")
          
      sys.stdout.flush()
      t_last = t_now
    yield item
  t_total = time.time() - t_start
  print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))


# Sample function for use with inception metrics
def sample(G, z_, y_, config, z=None, y=None, use_drs=False):
  # assert(use_drs ==  False)
  with torch.no_grad():
    z_.sample_()
    y_.sample_()
    zg, yg = z_, y_
    if torch.is_tensor(z):
      zg = z
    if y is not None:
      if torch.is_tensor(y):
        yg = y
      else:
        yg.fill_(y)
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (zg, G.shared(yg)))
    else:
      G_z = G(zg, G.shared(yg))
    return G_z, yg


def sample_drs(G, z_, y_, config, z=None, y=None, D=None,
               M_state=None, use_drs=True, use_deterministic=False):
  with torch.no_grad():
    z_.sample_()
    y_.sample_()
    zg, yg = z_, y_
    if torch.is_tensor(z):
      zg = z
    if y is not None:
      if torch.is_tensor(y):
        yg = y
      else:
        yg.fill_(y)
    if config['parallel']:
      G_z =  nn.parallel.data_parallel(G, (zg, G.shared(yg)))
    else:
      G_z = G(zg, G.shared(yg))
    if use_drs and (D is not None) and (M_state is not None):
      logits = D(G_z, yg)
      logits = logits.reshape(-1).cpu().numpy()
      batch_ratio = np.exp(logits)
      max_idx = np.argmax(batch_ratio)
      max_ratio = batch_ratio[max_idx]
      if max_ratio > M_state['M']:
        M_state['M'] = max_ratio
        M_state['logit'] = logits[max_idx]
      max_M, max_logit = M_state['M'], M_state['logit']
      # Calculate F_hat
      Fs = logits - max_logit - np.log(1 - np.exp(logits - max_logit - 1e-8))
      gamma = np.percentile(Fs, config['gamma_percentile'])
      F_hat = Fs - gamma
      acceptance_prob = 1./(1 + np.exp(-F_hat))  # sigmoid
      x_accept = []
      y_accept = []
      num_accepted = 0
      for idx, (sample, label) in enumerate(zip(G_z, yg)):
        probability = np.random.uniform(0, 1)
        # print ('prob : %.4f p : %.4f F_hat : %.4f F : %.4f'%(probability, acceptance_prob[idx], F_hat[idx], Fs[idx]))
        if (probability <= acceptance_prob[idx] or 
           (use_deterministic and acceptance_prob[idx] >= config['gamma_percentile'])):
          x_accept.append(sample)
          y_accept.append(label)
          num_accepted += 1
      G_z = torch.stack(x_accept) if num_accepted > 0 else []
      yg = torch.stack(y_accept) if num_accepted > 0 else []
    return G_z, yg


# Sample function for sample sheets
def sample_sheet(G, classes_per_sheet, num_classes, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number, z_=None,
                 sample=None, use_drs=False, drs_classes=[]):
  # drs_classes is not supported for sample_sheet for now
  # Prepare sample directory
  if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
    os.mkdir('%s/%s' % (samples_root, experiment_name))
  if not os.path.isdir(f'{samples_root}/{experiment_name}/{folder_number}'):
    os.mkdir(f'{samples_root}/{experiment_name}/{folder_number}')
  # loop over total number of sheets
  for i in range(num_classes // classes_per_sheet):
    ims = []
    y = torch.arange(i * classes_per_sheet, (i + 1) * classes_per_sheet, device='cuda')
    if sample is not None:
      classes = list(range(i * classes_per_sheet, (i + 1) * classes_per_sheet))
    for j in range(samples_per_class):
      if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
        z_.sample_()
      else:
        z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
      if G.fp16:
        z_ = z_.half()
      with torch.no_grad():
        if sample is None:
          if parallel:
            o = nn.parallel.data_parallel(G, (z_[:classes_per_sheet], G.shared(y)))
          else:
            o = G(z_[:classes_per_sheet], G.shared(y))
        else:
          # Use sample() if not None
          img_cls = {c: [] for c in classes}
          cnt_cls = {c: 0 for c in classes}
          while True:
            oa, ya = sample(z=z_[:classes_per_sheet], y=y, use_drs=use_drs)
            if len(oa) == 0:
              continue
            for image, label in zip(oa, ya):
              img_cls[label.cpu().item()].append(image)
              cnt_cls[label.cpu().item()] += 1
            if all(np.array(list(cnt_cls.values())) >= 1):
              break
            if (z_ is not None) and hasattr(z_, 'sample_') and classes_per_sheet <= z_.size(0):
              z_.sample_()
            else:
              z_ = torch.randn(classes_per_sheet, G.dim_z, device='cuda')
          o = torch.stack([img_cls[c][0] for c in classes], dim=0)
      ims += [o.data.cpu()]
    # This line should properly unroll the images
    out_ims = torch.stack(ims, 1).view(-1, ims[0].shape[1], ims[0].shape[2], 
                                       ims[0].shape[3]).data.float().cpu()
    # The path for the samples
    image_filename = f'{samples_root}/{experiment_name}/{folder_number}/samples{i}.jpg'
    torchvision.utils.save_image(out_ims, image_filename,
                                 nrow=samples_per_class, normalize=True)


# Sample pairs
def sample_pairs(G, G_batch_size, samples_per_class, parallel,
                 samples_root, experiment_name, folder_number='pairs', z_=None, y_=None):
  # Prepare sample directory
  if not os.path.isdir('%s/%s' % (samples_root, experiment_name)):
    os.mkdir('%s/%s' % (samples_root, experiment_name))
  if not os.path.isdir(f'{samples_root}/{experiment_name}/{folder_number}'):
    os.mkdir(f'{samples_root}/{experiment_name}/{folder_number}')
  # loop over batches
  ims = {0: [], 1: []}
  for i in range(max(1, math.ceil(1.0*samples_per_class/G_batch_size))):
    for j in [0, 1]:
      if (z_ is not None) and hasattr(z_, 'sample_') and G_batch_size <= z_.size(0):
        z_.sample_()
      else:
        z_ = torch.randn(G_batch_size, G.dim_z, device='cuda')
      y_.fill_(j)
      with torch.no_grad():
        if parallel:
          o = nn.parallel.data_parallel(G, (z_[:G_batch_size], G.shared(y_)))
        else:
          o = G(z_[:G_batch_size], G.shared(y_))
      ims[j] += [o.data.cpu()]
  ims = [torch.cat(ims[y], dim=0).data.float().cpu() for y in [0, 1]]
  for j in range(samples_per_class):
    image_filename = f'{samples_root}/{experiment_name}/{folder_number}/pair{j:04d}.jpg'
    out_im = []
    for y in [0, 1]:
      out_im.append(ims[y][j])
    torchvision.utils.save_image(torch.stack(out_im), image_filename, nrow=2, normalize=True)


# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
  lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
  return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))


# interp sheet function
# Supports full, class-wise and intra-class interpolation
def interp_sheet(G, num_per_sheet, num_midpoints, num_classes, parallel,
                 samples_root, experiment_name, folder_number, sheet_number=0,
                 fix_z=False, fix_y=False, device='cuda'):
  # Prepare zs and ys
  if fix_z: # If fix Z, only sample 1 z per row
    zs = torch.randn(num_per_sheet, 1, G.dim_z, device=device)
    zs = zs.repeat(1, num_midpoints + 2, 1).view(-1, G.dim_z)
  else:
    zs = interp(torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                torch.randn(num_per_sheet, 1, G.dim_z, device=device),
                num_midpoints).view(-1, G.dim_z)
  if fix_y: # If fix y, only sample 1 z per row
    ys = sample_1hot(num_per_sheet, num_classes)
    ys = G.shared(ys).view(num_per_sheet, 1, -1)
    ys = ys.repeat(1, num_midpoints + 2, 1).view(num_per_sheet * (num_midpoints + 2), -1)
  else:
    ys = interp(G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                G.shared(sample_1hot(num_per_sheet, num_classes)).view(num_per_sheet, 1, -1),
                num_midpoints).view(num_per_sheet * (num_midpoints + 2), -1)
  # Run the net--note that we've already passed y through G.shared.
  if G.fp16:
    zs = zs.half()
  with torch.no_grad():
    if parallel:
      out_ims = nn.parallel.data_parallel(G, (zs, ys)).data.cpu()
    else:
      out_ims = G(zs, ys).data.cpu()
  interp_style = '' + ('Z' if not fix_z else '') + ('Y' if not fix_y else '')
  image_filename = f'{samples_root}/{experiment_name}/{folder_number}/interp{interp_style}{sheet_number}.jpg'
  torchvision.utils.save_image(out_ims, image_filename,
                               nrow=num_midpoints + 2, normalize=True)


# Convenience debugging function to print out gradnorms and shape from each layer
# May need to rewrite this so we can actually see which parameter is which
def print_grad_norms(net):
    gradsums = [[float(torch.norm(param.grad).item()),
                 float(torch.norm(param).item()), param.shape]
                for param in net.parameters()]
    order = np.argsort([item[0] for item in gradsums])
    print(['%3.3e,%3.3e, %s' % (gradsums[item_index][0],
                                gradsums[item_index][1],
                                str(gradsums[item_index][2])) 
                              for item_index in order])


# Get singular values to log. This will use the state dict to find them
# and substitute underscores for dots.
def get_SVs(net, prefix):
  d = net.state_dict()
  return {('%s_%s' % (prefix, key)).replace('.', '_') :
            float(d[key].item())
            for key in d if 'sv' in key}


# Name an experiment based on its config
def name_from_config(config):
  name = '_'.join([
  item for item in [
  'Big%s' % config['which_train_fn'],
  config['dataset'],
  config['model'] if config['model'] != 'BigGAN' else None,
  'seed%d' % config['seed'],
  'Gch%d' % config['G_ch'],
  'Dch%d' % config['D_ch'],
  'Gd%d' % config['G_depth'] if config['G_depth'] > 1 else None,
  'Dd%d' % config['D_depth'] if config['D_depth'] > 1 else None,
  'bs%d' % config['batch_size'],
  'Gfp16' if config['G_fp16'] else None,
  'Dfp16' if config['D_fp16'] else None,
  'nDs%d' % config['num_D_steps'] if config['num_D_steps'] > 1 else None,
  'nDa%d' % config['num_D_accumulations'] if config['num_D_accumulations'] > 1 else None,
  'nGa%d' % config['num_G_accumulations'] if config['num_G_accumulations'] > 1 else None,
  'Glr%2.1e' % config['G_lr'],
  'Dlr%2.1e' % config['D_lr'],
  'GB%3.3f' % config['G_B1'] if config['G_B1'] !=0.0 else None,
  'GBB%3.3f' % config['G_B2'] if config['G_B2'] !=0.999 else None,
  'DB%3.3f' % config['D_B1'] if config['D_B1'] !=0.0 else None,
  'DBB%3.3f' % config['D_B2'] if config['D_B2'] !=0.999 else None,
  'Gnl%s' % config['G_nl'],
  'Dnl%s' % config['D_nl'],
  'Ginit%s' % config['G_init'],
  'Dinit%s' % config['D_init'],
  'G%s' % config['G_param'] if config['G_param'] != 'SN' else None,
  'D%s' % config['D_param'] if config['D_param'] != 'SN' else None,
  'Gattn%s' % config['G_attn'] if config['G_attn'] != '0' else None,
  'Dattn%s' % config['D_attn'] if config['D_attn'] != '0' else None,
  'Gortho%2.1e' % config['G_ortho'] if config['G_ortho'] > 0.0 else None,
  'Dortho%2.1e' % config['D_ortho'] if config['D_ortho'] > 0.0 else None,
  config['norm_style'] if config['norm_style'] != 'bn' else None,
  'cr' if config['cross_replica'] else None,
  'Gshared' if config['G_shared'] else None,
  'hier' if config['hier'] else None,
  'ema' if config['ema'] else None,
  config['name_suffix'] if config['name_suffix'] else None,
  ]
  if item is not None])
  # dogball
  if config['hashname']:
    return hashname(name)
  else:
    return name


# A simple function to produce a unique experiment name from the animal hashes.
def hashname(name):
  h = hash(name)
  a = h % len(animal_hash.a)
  h = h // len(animal_hash.a)
  b = h % len(animal_hash.b)
  h = h // len(animal_hash.c)
  c = h % len(animal_hash.c)
  return animal_hash.a[a] + animal_hash.b[b] + animal_hash.c[c]


# Get GPU memory, -i is the index
def query_gpu(indices):
  os.system('nvidia-smi -i 0 --query-gpu=memory.free --format=csv')


# Convenience function to count the number of parameters in a module
def count_parameters(module):
  print('Number of parameters: {}'.format(
    sum([p.data.nelement() for p in module.parameters()])))

   
# Convenience function to sample an index, not actually a 1-hot
def sample_1hot(batch_size, num_classes, device='cuda'):
  return torch.randint(low=0, high=num_classes, size=(batch_size,),
          device=device, dtype=torch.int64, requires_grad=False)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
  # Init the params of the distribution
  def init_distribution(self, dist_type, **kwargs):    
    self.dist_type = dist_type
    self.dist_kwargs = kwargs
    if self.dist_type == 'normal':
      self.mean, self.var = kwargs['mean'], kwargs['var']
    elif self.dist_type == 'categorical':
      self.num_categories = kwargs['num_categories']

  def sample_(self):
    if self.dist_type == 'normal':
      self.normal_(self.mean, self.var)
    elif self.dist_type == 'categorical':
      self.random_(0, self.num_categories)    
    # return self.variable
    
  # Silly hack: overwrite the to() method to wrap the new object
  # in a distribution as well
  def to(self, *args, **kwargs):
    new_obj = Distribution(self)
    new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
    new_obj.data = super().to(*args, **kwargs)    
    return new_obj


# Convenience function to prepare a z and y vector
def prepare_z_y(G_batch_size, dim_z, nclasses, device='cuda', 
                fp16=False, z_var=1.0):
  z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
  z_.init_distribution('normal', mean=0, var=z_var)
  z_ = z_.to(device, torch.float16 if fp16 else torch.float32)
  
  if fp16:
    z_ = z_.half()

  y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
  y_.init_distribution('categorical', num_categories=nclasses)
  y_ = y_.to(device, torch.int64)
  return z_, y_


def initiate_standing_stats(net):
  for module in net.modules():
    if hasattr(module, 'accumulate_standing'):
      module.reset_stats()
      module.accumulate_standing = True


def accumulate_standing_stats(net, z, y, nclasses, num_accumulations=16):
  initiate_standing_stats(net)
  net.train()
  for i in range(num_accumulations):
    with torch.no_grad():
      z.normal_()
      y.random_(0, nclasses)
      x = net(z, net.shared(y)) # No need to parallelize here unless using syncbn
  # Set to eval mode
  net.eval() 


# This version of Adam keeps an fp32 copy of the parameters and
# does all of the parameter updates in fp32, while still doing the
# forwards and backwards passes using fp16 (i.e. fp16 copies of the
# parameters and fp16 activations).
#
# Note that this calls .float().cuda() on the params.
import math
from torch.optim.optimizer import Optimizer
class Adam16(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0):
    defaults = dict(lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay)
    params = list(params)
    super(Adam16, self).__init__(params, defaults)
      
  # Safety modification to make sure we floatify our state
  def load_state_dict(self, state_dict):
    super(Adam16, self).load_state_dict(state_dict)
    for group in self.param_groups:
      for p in group['params']:
        self.state[p]['exp_avg'] = self.state[p]['exp_avg'].float()
        self.state[p]['exp_avg_sq'] = self.state[p]['exp_avg_sq'].float()
        self.state[p]['fp32_p'] = self.state[p]['fp32_p'].float()

  def step(self, closure=None):
    """Performs a single optimization step.
    Arguments:
      closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
    """
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      for p in group['params']:
        if p.grad is None:
          continue
          
        grad = p.grad.data.float()
        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = 0
          # Exponential moving average of gradient values
          state['exp_avg'] = grad.new().resize_as_(grad).zero_()
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()
          # Fp32 copy of the weights
          state['fp32_p'] = p.data.float()

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        state['step'] += 1

        if group['weight_decay'] != 0:
          grad = grad.add(group['weight_decay'], state['fp32_p'])

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(1 - beta1, grad)
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

        denom = exp_avg_sq.sqrt().add_(group['eps'])

        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']
        step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
      
        state['fp32_p'].addcdiv_(-step_size, exp_avg, denom)
        p.data = state['fp32_p'].half()

    return loss


class ConstEmbedding(nn.Module):
  def __init__(self, n_classes=1000, values=None):
    super(ConstEmbedding, self).__init__()
    self.embed = nn.Embedding(n_classes, 1)
    self.entropy = 0.

    # init weights
    if values is not None:
      self.embed.weight.data.copy_(torch.from_numpy(np.log(values)).view(n_classes, 1))
      self.entropy = -np.sum(values * np.log(values))

  def forward(self, y):
    return self.embed(y)


def prepare_logpy(n_classes, uniform, dataset, data_root, device='cuda', fp16=False):
  dataset = dataset.strip('_hdf5')
  if uniform:
    values = np.ones((n_classes, 1)) * (1. / n_classes)
  else:
    values = np.load(os.path.join(data_root, dataset+'_labels.npz'))['prior']
  embed = ConstEmbedding(n_classes, values)
  embed = embed.to(device, torch.float16 if fp16 else torch.float32)
  if fp16:
    embed = embed.half()
  return embed


def print_config(parser, config):
  message = ''
  message += '---------------------------- Config -----------------------------\n'
  for k, v in sorted(config.items()):
    comment = ''
    default = parser.get_default(k)
    if v != default:
      comment = '\t[default: %s]' % str(default)
    message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
  message += '----------------------------- End -------------------------------'
  print(message)

  # save to the disk
  experiment_name = (config['experiment_name'] if config['experiment_name']
                     else name_from_config(config))
  experiment_dir = '%s/%s/' % (config['logs_root'], experiment_name)
  if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
  file_name = os.path.join(experiment_dir, 'config.txt')
  with open(file_name, 'wt') as cfg_file:
    cfg_file.write(message)
    cfg_file.write('\n')

  # save command to disk
  file_name = os.path.join(experiment_dir, 'command.txt')
  with open(file_name, 'wt') as cmd_file:
    cmd_file.write('(%s) ' % os.getenv('HOSTNAME'))
    if os.getenv('CUDA_VISIBLE_DEVICES'):
      cmd_file.write('CUDA_VISIBLE_DEVICES=%s ' % os.getenv('CUDA_VISIBLE_DEVICES'))
    cmd_file.write(' '.join(sys.argv))
    cmd_file.write('\n')


def append_npy(filename, item):
  if not isinstance(item, np.ndarray):
    item = np.array(item)
  item = np.expand_dims(item, axis=0)
  if os.path.exists(filename):
    arr = np.load(filename)
    try:
      arr = np.concatenate([arr, item], axis=0)
    except:
      print(f'Numpy array dimension mismatch: arr.shape is {arr.shape}, item.shape is {item.shape}.')
      print(f'Backing up {filename} and creating a new file...')
      shutil.copyfile(filename, filename+f'.backup_{datetime.datetime.now()}.npy')
      arr = item
  else:
    arr = item
  np.save(filename, arr)


def make_linear(which_linear, input_dim, output_dims, bias=True, nl='none'):
  # nl is only applied after the last linear, if any
  n_layers = len(output_dims)
  blocks = []
  nf_prev = input_dim
  for i in range(n_layers - 1):
    nf = output_dims[i]
    blocks += [which_linear(nf_prev, nf), nn.Tanh()]
    nf_prev = nf
  if n_layers > 0:
    blocks += [which_linear(nf_prev, output_dims[-1], bias=bias), activation_dict[nl]]
  linear = nn.Sequential(*blocks) if n_layers > 0 else None
  return linear


def str2list(attr_bins):
  assert (isinstance(attr_bins, str))
  attr_bins = attr_bins.strip()
  if attr_bins.endswith(('.npy', '.npz')):
    attr_bins = np.load(attr_bins)
  else:
    assert (attr_bins.startswith('[') and attr_bins.endswith(']'))
    # attr_bins = np.array(ast.literal_eval(attr_bins))
    attr_bins = ast.literal_eval(attr_bins)
  return attr_bins


# def flip_binary_label(y, rate=0):
#   num_flip = max(1, int(y.numel()*rate))
#   idx = np.random.choice(y.numel(), num_flip, replace=False)
#   y_comp = 1 - y
#   y[idx] = y_comp[idx]
#   return y

@torch.no_grad()
def flip_labels(y, n_classes, rate=0):
  if rate <= 0:
    return y
  num_flip = max(1, int(y.numel()*rate))
  idx = np.random.choice(y.numel(), num_flip, replace=False)
  y[idx] = y[idx].random_(0, n_classes)
  return y

# @torch.no_grad()
# def balance_binary_labels(y, rate=0):
#   n0 = torch.count_nonzero(y == 0)
#   n1 = torch.count_nonzero(y == 1)
#   if n0 > n1:
#     # flip a portion of 0 to 1
#     m = (n0 - n1)/2
#     num_flip = max(1, int(m*rate))
#     idx = np.random.choice((y==0).nonzero().view(-1).cpu().numpy(), num_flip, replace=False)
#     y[idx] = 1
#   else:
#     # flip a portion of 1 to 0
#     m = (n1 - n0)/2
#     num_flip = max(1, int(m*rate))
#     idx = np.random.choice((y==1).nonzero().view(-1).cpu().numpy(), num_flip, replace=False)
#     y[idx] = 0
#   return y

import ot
@torch.no_grad()
def balance_labels(y, n_classes, rate=0, target=[]):
  ns = np.histogram(y.cpu().numpy(), bins=range(n_classes+1))[0]
  if not target:
    nt = np.ones(n_classes) * y.numel() / n_classes
  else:
    if isinstance(target, list):
      target = np.array(target)
    nt = target / np.sum(target) * y.numel()
  M = np.ones((n_classes, n_classes)) - np.eye(n_classes)
  p = ot.emd(ns, nt, M)
  num_flip = (p * M * rate).astype(np.int64)
  i, j = num_flip.nonzero()
  for i_, j_ in zip(i, j):
    # flip i_ to j_
    idx_valid = torch.nonzero(y == i_).view(-1).cpu().numpy()
    idx = np.random.choice(idx_valid, num_flip[i_, j_], replace=False)
    y[idx] = j_
  return y

@torch.no_grad()
def truncate_labels(y, n_classes, rate=0, target=[]):
  # Truncates to minimum number of samples, set a portion to zero
  # Returns weight
  ns = np.histogram(y.cpu().numpy(), bins=range(n_classes+1))[0]
  if not target:
    nt = np.ones(n_classes)
  elif isinstance(target, list):
    nt = np.array(target)
  r = ns / nt
  nt_new = nt * r.min()
  weight = torch.ones_like(y)
  for i_ in range(n_classes):
    idx_valid = torch.nonzero(y == i_).view(-1)
    num_flip = int((idx_valid.numel() - nt_new[i_]) * rate)
    if num_flip > 0:
      idx = np.random.choice(idx_valid.cpu().numpy(), num_flip, replace=False)
      weight[idx] = 0.
  return weight

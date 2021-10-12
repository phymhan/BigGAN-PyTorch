''' train_fns.py
Functions for the main loop of training different conditional image models
'''
import torch
import torch.nn as nn
import torchvision
import os

import utils
import losses
import torch.nn.functional as F
import numpy as np
import pdb
st = pdb.set_trace

# Dummy training function for debugging
def dummy_training_function():
  def train(x, y):
    return {}
  return train


def GAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  def train(x, y):
    # If y is tuple, y is (y, weight)
    if isinstance(y, tuple):
      y, weight = y
      weight = torch.split(weight, config['batch_size'])
    else:
      weight = None
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])

        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        w_real = weight[counter] if weight is not None else None
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real, None, w_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = generator_loss(D_fake) / float(config['num_G_accumulations'])
      G_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
            'D_loss_real': float(D_loss_real.item()),
            'D_loss_fake': float(D_loss_fake.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train


def fairGAN_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  if config['fair_which_div'] in ['wgan', 'gan']:
    z0, y0 = utils.prepare_z_y(z_.size(0), z_.size(1), config['n_classes'],
                              device='cuda', fp16=config['G_fp16'])
    y0.fill_(0)
    z1, y1 = utils.prepare_z_y(z_.size(0), z_.size(1), config['n_classes'],
                              device='cuda', fp16=config['G_fp16'])
    y1.fill_(1)

  def train(x, y):
    # If y is tuple, y is (y, weight)
    if isinstance(y, tuple):
      y, weight = y
      weight = torch.split(weight, config['batch_size'])
    else:
      weight = None
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    batch_size = config['batch_size']
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, cls_out = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])
         
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        w_real = weight[counter] if weight is not None else None
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real, None, w_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

        cls_loss = 0.
        if config['fair_which_div'] in ['wgan', 'gan']:
          # Sample y=0 and y=1 batches
          z0.sample_()
          # y_.fill_(0)
          _, f_0 = GD(z0, y0)
          z1.sample_()
          # y_.fill_(1)
          _, f_1 = GD(z1, y1)
          if config['fair_which_div'] == 'wgan':
            cls_loss = torch.mean(f_1) - torch.mean(f_0)
          else:  # config['fair_which_div'] == 'gan':
            cls_loss_0, cls_loss_1 = discriminator_loss(f_1, f_0)
            cls_loss = cls_loss_0 + cls_loss_1
        elif config['fair_which_div'] == 'jsd':
          cls_loss = F.cross_entropy(cls_out[:batch_size], y_)
        D_loss += cls_loss * config['fair_weight_div'] / float(config['num_D_accumulations'])

        D_loss.backward()
        counter += 1

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, cls_out = GD(z_, y_, train_G=True, split_D=config['split_D'])
      G_loss = generator_loss(D_fake) / float(config['num_G_accumulations'])

      div_loss = 0.
      if config['fair_which_div'] in ['wgan', 'gan']:
        z0.sample_()
        # y_.fill_(0)
        _, f_0 = GD(z0, y0)
        z1.sample_()
        # y_.fill_(1)
        _, f_1 = GD(z1, y1)
        if config['fair_which_div'] == 'wgan':
          div_loss = torch.mean(f_0) - torch.mean(f_1)
        else:  # config['fair_which_div'] == 'gan':
          cls_loss_0, cls_loss_1 = discriminator_loss(f_1, f_0)
          div_loss = -(cls_loss_0 + cls_loss_1)
      elif config['fair_which_div'] == 'jsd':
        cls_loss = F.cross_entropy(cls_out, y_)
        div_loss = -cls_loss
      G_loss += div_loss * config['fair_weight_div'] / float(config['num_G_accumulations'])

      G_loss.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'cls_loss': cls_loss if isinstance(cls_loss, float) else float(cls_loss.item()),
           'div_loss': div_loss if isinstance(div_loss, float) else float(div_loss.item())}
    # Return G's loss and the components of D's loss.
    return out
  return train


def DRS_training_function(G, D, GD, z_, y_, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss = losses.loss_hinge_dis
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss = losses.loss_dcgan_dis
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss = losses.loss_bce_dis
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss = losses.loss_lsgan_dis
  def train(x, y):
    # If y is tuple, y is (y, weight)
    if isinstance(y, tuple):
      y, weight = y
      weight = torch.split(weight, config['batch_size'])
    else:
      weight = None
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real = GD(z_[:config['batch_size']], y_[:config['batch_size']], 
                            x[counter], y[counter], train_G=False, 
                            split_D=config['split_D'])

        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        w_real = weight[counter] if weight is not None else None
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real, None, w_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        D_loss.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    out = {'D_loss_real': float(D_loss_real.item()), 'D_loss_fake': float(D_loss_fake.item())}
    return out
  return train


def hybridcGAN_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['f_div_loss'] == 'revkl':
    f_div_loss = losses.f_div_loss_revkl
  elif config['f_div_loss'] == 'hinge1':
    f_div_loss = losses.f_div_loss_hinge1
  elif config['f_div_loss'] == 'hinge0':
    f_div_loss = losses.f_div_loss_hinge0
  elif config['f_div_loss'] == 'proj':
    f_div_loss = losses.f_div_loss_proj
  elif config['f_div_loss'] == 'kl':
    f_div_loss = losses.f_div_loss_kl
  elif config['f_div_loss'] == 'pearson':
    f_div_loss = losses.f_div_loss_pearson
  elif config['f_div_loss'] == 'squared':
    f_div_loss = losses.f_div_loss_squared
  elif config['f_div_loss'] == 'jsd':
    f_div_loss = losses.f_div_loss_jsd
  elif config['f_div_loss'] == 'gan':
    f_div_loss = losses.f_div_loss_gan
  else:
    raise NotImplementedError

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0

    mi_scale = 1
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, _, _, _ = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                     x[counter], y[counter],
                                     train_G=False, split_D=config['split_D'])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if config['loss_type'] == 'hybrid':
          ac_fake = ac[:batch_size]
          ac_real = ac[batch_size:]
          tac_fake = tac[:batch_size]
          tac_real = tac[batch_size:]
          D_fake = D_fake + ac_fake[range(y_.size(0)), y_] - tac_fake[range(y_.size(0)), y_]
          D_real = D_real + ac_real[range(y[counter].size(0)), y[counter]] - tac_real[range(y[counter].size(0)), y[counter]]
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        C_loss = 0.

        # AC loss
        if config['loss_type'] == 'AC' or config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid':
          C_loss = F.cross_entropy(ac[batch_size:], y[counter])
          if config['train_AC_on_fake']:
            C_loss += F.cross_entropy(ac[:batch_size], y_)
        
        # TAC loss
        if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid':
          C_loss += F.cross_entropy(tac[:batch_size], y_)
        
        C_loss = C_loss / float(config['num_D_accumulations'])
        if config['mi_weight_decay'] < 1:
          mi_scale = mi_scale * config['mi_weight_decay']
        D_loss_full = D_loss + C_loss * config['AC_weight'] * mi_scale
        D_loss_full.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, _, _, _ = GD(z_, y_, train_G=True, split_D=config['split_D'])

      # f-div
      f_loss = 0.
      if config['loss_type'] == 'hybrid':
        D_fake = D_fake + ac[range(y_.size(0)), y_] - tac[range(y_.size(0)), y_]
      if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN':
        f_loss = f_div_loss(ac, tac, y_)
      if config['loss_type'] == 'AC':
        f_loss = F.cross_entropy(ac, y_)
      G_loss = generator_loss(D_fake)
      G_loss_full = (G_loss + f_loss * config['AC_weight'] * mi_scale) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item())}
    if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid' or config['loss_type'] == 'AC':
      out['C_loss'] = float(C_loss.item())
      if config['loss_type'] != 'hybrid':
        out['f_loss'] = float(f_loss.item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def hybridcGAN_adaptive_scalar_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['f_div_loss'] == 'revkl':
    f_div_loss = losses.f_div_loss_revkl
  elif config['f_div_loss'] == 'hinge1':
    f_div_loss = losses.f_div_loss_hinge1
  elif config['f_div_loss'] == 'hinge0':
    f_div_loss = losses.f_div_loss_hinge0
  elif config['f_div_loss'] == 'proj':
    f_div_loss = losses.f_div_loss_proj
  elif config['f_div_loss'] == 'kl':
    f_div_loss = losses.f_div_loss_kl
  elif config['f_div_loss'] == 'pearson':
    f_div_loss = losses.f_div_loss_pearson
  elif config['f_div_loss'] == 'squared':
    f_div_loss = losses.f_div_loss_squared
  elif config['f_div_loss'] == 'jsd':
    f_div_loss = losses.f_div_loss_jsd
  elif config['f_div_loss'] == 'gan':
    f_div_loss = losses.f_div_loss_gan
  else:
    raise NotImplementedError

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, _, _, _ = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                     x[counter], y[counter],
                                     train_G=False, split_D=config['split_D'])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if config['loss_type'] == 'hybrid':
          ac_fake = ac[:batch_size]
          ac_real = ac[batch_size:]
          tac_fake = tac[:batch_size]
          tac_real = tac[batch_size:]
          if config['hybrid_noisy']:
            D_fake = D_fake.view(-1, 1)
            D_real = D_real.view(-1, 1)
          D_fake = D_fake + ac_fake[range(y_.size(0)), y_] - tac_fake[range(y_.size(0)), y_]
          D_real = D_real + ac_real[range(y[counter].size(0)), y[counter]] - tac_real[range(y[counter].size(0)), y[counter]]
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        C_loss = 0.

        # AC loss
        if config['loss_type'] == 'AC' or config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid':
          C_loss = F.cross_entropy(ac[batch_size:], y[counter])
          if config['train_AC_on_fake']:
            C_loss += F.cross_entropy(ac[:batch_size], y_)
        
        # TAC loss
        if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid':
          C_loss += F.cross_entropy(tac[:batch_size], y_)
        
        if config['add_weight_penalty']:
          # penalty = 0.5*D.scalar_sigmoid - F.logsigmoid(D.scalar_sigmoid)
          penalty = D.scalar_sigmoid * config['lambda_penalty_weight']
        else:
          penalty = 0.
        
        C_loss = C_loss / float(config['num_D_accumulations'])
        # D_loss_full = D_loss * torch.sigmoid(D.scalar_sigmoid) + C_loss * config['AC_weight'] * (1. - torch.sigmoid(D.scalar_sigmoid)) + penalty
        D_loss_full = D_loss + C_loss * config['AC_weight'] * torch.exp(-D.scalar_sigmoid) + penalty
        D_loss_full.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      D.scalar_sigmoid.grad *= config['lambda_lr']
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, _, _, _ = GD(z_, y_, train_G=True, split_D=config['split_D'])

      # f-div
      f_loss = 0.
      if config['loss_type'] == 'hybrid':
        D_fake = D_fake + ac[range(y_.size(0)), y_] - tac[range(y_.size(0)), y_]
      if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN':
        f_loss = f_div_loss(ac, tac, y_)
      if config['loss_type'] == 'AC':
        f_loss = F.cross_entropy(ac, y_)
      G_loss = generator_loss(D_fake)
      # G_loss_full = (G_loss*torch.sigmoid(D.scalar_sigmoid) + f_loss * config['AC_weight']) / float(config['num_G_accumulations'])
      G_loss_full = (G_loss + f_loss * config['AC_weight']) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item())}
    if config['loss_type'] == 'TAC' or config['loss_type'] == 'fcGAN' or config['loss_type'] == 'hybrid' or config['loss_type'] == 'AC':
      out['C_loss'] = float(C_loss.item())
      if config['loss_type'] != 'hybrid':
        out['f_loss'] = float(f_loss.item())
    out['weight_x'] = float(torch.exp(-D.scalar_sigmoid).item())
    out['weight_y'] = float(1.-torch.exp(-D.scalar_sigmoid).item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def hybridcGAN_adaptive_amortised_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['f_div_loss'] == 'revkl':
    f_div_loss = losses.f_div_loss_revkl
  elif config['f_div_loss'] == 'hinge1':
    f_div_loss = losses.f_div_loss_hinge1
  elif config['f_div_loss'] == 'hinge0':
    f_div_loss = losses.f_div_loss_hinge0
  elif config['f_div_loss'] == 'proj':
    f_div_loss = losses.f_div_loss_proj
  elif config['f_div_loss'] == 'kl':
    f_div_loss = losses.f_div_loss_kl
  elif config['f_div_loss'] == 'pearson':
    f_div_loss = losses.f_div_loss_pearson
  elif config['f_div_loss'] == 'squared':
    f_div_loss = losses.f_div_loss_squared
  else:
    raise NotImplementedError
  
  if config['lambda_clip'] == 'range' and len(config['clip_value']) == 1:
    config['clip_value'] = [-abs(config['clip_value']), abs(config['clip_value'])]

  def train(x, y):
    assert(config['use_hybrid'])
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0

    grad_norm_w_ = 0.
    grad_norm_b_ = 0.
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, wx, _, _ = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                               x[counter], y[counter],
                                               train_G=False, split_D=config['split_D'])
        # weight for real/fake loss
        if config['gated_lambda']:
          wx_fake = torch.sigmoid(wx[:batch_size])
          wx_real = torch.sigmoid(wx[batch_size:])
        else:
          wx_fake = torch.exp(-wx[:batch_size])
          wx_real = torch.exp(-wx[batch_size:])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if config['use_hybrid']:
          ac_fake = ac[:batch_size]
          ac_real = ac[batch_size:]
          tac_fake = tac[:batch_size]
          tac_real = tac[batch_size:]
          D_fake = D_fake + ac_fake[range(y_.size(0)), y_] - tac_fake[range(y_.size(0)), y_]
          D_real = D_real + ac_real[range(y[counter].size(0)), y[counter]] - tac_real[range(y[counter].size(0)), y[counter]]
        if config['gated_lambda']:
          D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real, 1.-wx_fake, 1.-wx_real)
        else:
          D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

        # AC loss
        C_loss = torch.mean(F.cross_entropy(ac[batch_size:], y[counter], reduction='none') * wx_real.view(-1))
        
        # TAC loss
        C_loss += torch.mean(F.cross_entropy(tac[:batch_size], y_, reduction='none') * wx_fake.view(-1))
        
        C_loss = C_loss / float(config['num_D_accumulations'])
        if config['add_weight_penalty']:
          if config['gated_lambda']:
            # penalty = -torch.mean(torch.log(wx_fake*(1.-wx_fake)+1e-8)) - torch.mean(torch.log(wx_real*(1.-wx_real)+1e-8))
            penalty = -torch.mean(F.logsigmoid(wx_fake) + F.logsigmoid(-wx_fake) + F.logsigmoid(wx_real) + F.logsigmoid(-wx_real))
          else:
            penalty = torch.mean(wx)
          penalty *= config['lambda_penalty_weight']
        else:
          penalty = 0.
        D_loss_full = D_loss + C_loss * config['AC_weight'] + penalty / float(config['num_D_accumulations'])
        D_loss_full.backward()
        counter += 1

      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      with torch.no_grad():
        grad_norm_w_ = torch.max(D.linear_wx[-2].weight.grad.data).cpu().item()
        grad_norm_b_ = torch.max(D.linear_wx[-2].bias.grad.data).cpu().item()
      # Clip grad of linear_wx
      if config['lambda_grad_clip'] == 'value':
        nn.utils.clip_grad_value_(D.linear_wx.parameters(), config['clip_value'][0])
      elif config['lambda_grad_clip'] == 'norm':
        nn.utils.clip_grad_norm_(D.linear_wx.parameters(), config['clip_value'][0])
      if config['lambda_lr'] != 1:
        for p in D.linear_wx.parameters():
          p.grad *= config['lambda_lr']
      D.optim.step()

      # Clip linear_wx
      if config['lambda_clip'] != 'none':  # lambda_clip == 'value' or 'range'
        for p in D.linear_wx.parameters():
          p.data.clamp_(min=config['clip_value'][0], max=config['clip_value'][1])
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, wx, wp, wq = GD(z_, y_, train_G=True, split_D=config['split_D'])
      if config['gated_lambda']:
        wx = torch.sigmoid(wx).detach()
      else:
        wx = torch.exp(-wx).detach()

      # f-div
      if config['use_hybrid']:
        D_fake = D_fake + ac[range(y_.size(0)), y_] - tac[range(y_.size(0)), y_]
        f_loss = 0.
      else:
        f_loss = f_div_loss(ac, tac, y_, wx)
      if config['gated_lambda']:
        G_loss = generator_loss(D_fake, 1.-wx)
      else:
        G_loss = generator_loss(D_fake)
      G_loss_full = (G_loss + f_loss*config['AC_weight']) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'C_loss': float(C_loss.item())}
    out['weight_real'] = float(torch.mean(wx_real).item())
    out['weight_fake'] = float(torch.mean(wx_fake).item())
    out['grad_norm_w'] = grad_norm_w_
    out['grad_norm_b'] = grad_norm_b_
    if not config['use_hybrid']:
      out['f_loss'] = float(f_loss.item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def hybridcGAN_interp_amortised_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['f_div_loss'] == 'revkl':
    f_div_loss = losses.f_div_loss_revkl
  elif config['f_div_loss'] == 'hinge1':
    f_div_loss = losses.f_div_loss_hinge1
  elif config['f_div_loss'] == 'hinge0':
    f_div_loss = losses.f_div_loss_hinge0
  elif config['f_div_loss'] == 'proj':
    f_div_loss = losses.f_div_loss_proj
  elif config['f_div_loss'] == 'kl':
    f_div_loss = losses.f_div_loss_kl
  elif config['f_div_loss'] == 'pearson':
    f_div_loss = losses.f_div_loss_pearson
  elif config['f_div_loss'] == 'squared':
    f_div_loss = losses.f_div_loss_squared
  else:
    raise NotImplementedError

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, wx, wp, wq = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                     x[counter], y[counter],
                                     train_G=False, split_D=config['split_D'])
        wx = wx.view(-1)
        # weight for real/fake loss
        wx_fake = torch.sigmoid(wx[:batch_size])
        wx_real = torch.sigmoid(wx[batch_size:])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations

        ac_fake = ac[:batch_size]
        ac_real = ac[batch_size:]
        tac_fake = tac[:batch_size]
        tac_real = tac[batch_size:]
        D_fake = D_fake + 2 * (1-wx_fake) * (ac_fake[range(y_.size(0)), y_] - tac_fake[range(y_.size(0)), y_])
        D_real = D_real + 2 * (1-wx_real) * (ac_real[range(y[counter].size(0)), y[counter]] - tac_real[range(y[counter].size(0)), y[counter]])
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

        # AC loss
        C_loss = torch.mean(F.cross_entropy(ac[batch_size:], y[counter], reduction='none') * wx_real * 2)
        
        # TAC loss
        C_loss += torch.mean(F.cross_entropy(tac[:batch_size], y_, reduction='none') * wx_fake * 2)
        
        C_loss = C_loss / float(config['num_D_accumulations'])
        if config['add_weight_penalty']:
          penalty = -0.5*(torch.mean(torch.log(wx_fake*(1.-wx_fake)+1e-8)) + torch.mean(torch.log(wx_real*(1.-wx_real)+1e-8)))
          penalty = penalty * config['lambda_penalty_weight']
        else:
          penalty = 0.
        D_loss_full = D_loss + C_loss * config['AC_weight'] + penalty / float(config['num_D_accumulations'])
        D_loss_full.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      for p in D.linear_wx.parameters():
        p.grad *= config['lambda_lr']
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, wx, wp, wq = GD(z_, y_, train_G=True, split_D=config['split_D'])
      wx = torch.sigmoid(wx.view(-1)).detach()
      # wx = torch.exp(-wx.detach())

      # f-div
      D_fake = D_fake + 2 * (1-wx) * (ac[range(y_.size(0)), y_] - tac[range(y_.size(0)), y_])
      f_loss = f_div_loss(ac, tac, y_, wx * 2)
      G_loss = generator_loss(D_fake)
      G_loss_full = (G_loss + f_loss*config['AC_weight']) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'C_loss': float(C_loss.item())}
    out['weight_real'] = float(torch.mean(wx_real).item())
    out['weight_fake'] = float(torch.mean(wx_fake).item())
    if not config['use_hybrid']:
      out['f_loss'] = float(f_loss.item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def hybridcGAN_interp_scalar_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['f_div_loss'] == 'revkl':
    f_div_loss = losses.f_div_loss_revkl
  elif config['f_div_loss'] == 'hinge1':
    f_div_loss = losses.f_div_loss_hinge1
  elif config['f_div_loss'] == 'hinge0':
    f_div_loss = losses.f_div_loss_hinge0
  elif config['f_div_loss'] == 'proj':
    f_div_loss = losses.f_div_loss_proj
  elif config['f_div_loss'] == 'kl':
    f_div_loss = losses.f_div_loss_kl
  elif config['f_div_loss'] == 'pearson':
    f_div_loss = losses.f_div_loss_pearson
  elif config['f_div_loss'] == 'squared':
    f_div_loss = losses.f_div_loss_squared
  else:
    raise NotImplementedError

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
    
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, wx, wp, wq = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                     x[counter], y[counter],
                                     train_G=False, split_D=config['split_D'])
        # weight for real/fake loss
        # wx_fake = torch.sigmoid(wx[:batch_size])
        # wx_real = torch.sigmoid(wx[batch_size:])
        lambda_mi = torch.sigmoid(D.scalar_sigmoid)
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations

        ac_fake = ac[:batch_size]
        ac_real = ac[batch_size:]
        tac_fake = tac[:batch_size]
        tac_real = tac[batch_size:]
        D_fake = D_fake + 2 * (1-lambda_mi) * (ac_fake[range(y_.size(0)), y_] - tac_fake[range(y_.size(0)), y_])
        D_real = D_real + 2 * (1-lambda_mi) * (ac_real[range(y[counter].size(0)), y[counter]] - tac_real[range(y[counter].size(0)), y[counter]])
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])

        # AC loss
        C_loss = F.cross_entropy(ac[batch_size:], y[counter]) * lambda_mi * 2
        
        # TAC loss
        C_loss += F.cross_entropy(tac[:batch_size], y_) * lambda_mi * 2
        
        C_loss = C_loss / float(config['num_D_accumulations'])
        if config['add_weight_penalty']:
          # penalty = -0.5*(torch.mean(torch.log(wx_fake*(1.-wx_fake)+1e-8)) + torch.mean(torch.log(wx_real*(1.-wx_real)+1e-8)))
          penalty = -0.5*(F.logsigmoid(D.scalar_sigmoid) + F.logsigmoid(-D.scalar_sigmoid))
          penalty = penalty * config['lambda_penalty_weight']
        else:
          penalty = 0.
        D_loss_full = D_loss + C_loss * config['AC_weight'] + penalty / float(config['num_D_accumulations'])
        D_loss_full.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.scalar_sigmoid.grad *= config['lambda_lr']
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, wx, wp, wq = GD(z_, y_, train_G=True, split_D=config['split_D'])
      lambda_mi = torch.sigmoid(D.scalar_sigmoid.detach())
      # wx = torch.sigmoid(wx).detach()
      # wx = torch.exp(-wx.detach())

      # f-div
      D_fake = D_fake + 2 * (1-lambda_mi) * (ac[range(y_.size(0)), y_] - tac[range(y_.size(0)), y_])
      f_loss = f_div_loss(ac, tac, y_) * lambda_mi * 2
      G_loss = generator_loss(D_fake)
      G_loss_full = (G_loss + f_loss*config['AC_weight']) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'C_loss': float(C_loss.item())}
    out['lambda_mi'] = float(lambda_mi.item())
    if not config['use_hybrid']:
      out['f_loss'] = float(f_loss.item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def naive_hybridcGAN_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  f_div_loss = losses.f_div_loss_revkl

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        D_fake, D_real, ac, tac, out_pd, _, _ = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                     x[counter], y[counter],
                                     train_G=False, split_D=config['split_D'])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        C_loss = (F.cross_entropy(ac[batch_size:], y[counter]) + 
          F.cross_entropy(tac[:batch_size], y_)) / float(config['num_D_accumulations'])
        pD_loss_real, pD_loss_fake = discriminator_loss(out_pd[:batch_size], out_pd[batch_size:])
        pD_loss = (pD_loss_real + pD_loss_fake) / float(config['num_D_accumulations'])
        D_loss_full = D_loss + C_loss * config['AC_weight'] + pD_loss
        D_loss_full.backward()
        counter += 1
      
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      D_fake, ac, tac, out_pd, _, _ = GD(z_, y_, train_G=True, split_D=config['split_D'])

      # f-div
      f_loss = f_div_loss(ac, tac, y_)
      G_loss = generator_loss(D_fake)
      pG_loss = generator_loss(out_pd)
      G_loss_full = (G_loss + f_loss * config['AC_weight'] + pG_loss) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item())}
    out['C_loss'] = float(C_loss.item())
    out['f_loss'] = float(f_loss.item())
    # Return G's loss and the components of D's loss.
    return out
  return train


def fcGAN_training_function(G, D, GD, z_, y_, logpy, logqy, ema, state_dict, config):
  if config['GAN_loss'] == 'hinge':
    discriminator_loss, generator_loss = losses.loss_hinge_dis, losses.loss_hinge_gen
  elif config['GAN_loss'] == 'dcgan':
    discriminator_loss, generator_loss = losses.loss_dcgan_dis, losses.loss_dcgan_gen
  elif config['GAN_loss'] == 'vanilla':
    discriminator_loss, generator_loss = losses.loss_bce_dis, losses.loss_bce_gen
  elif config['GAN_loss'] == 'lsgan':
    discriminator_loss, generator_loss = losses.loss_lsgan_dis, losses.loss_lsgan_gen
  else:
    raise NotImplementedError
  if config['MI_loss'] == 'hinge':
    discriminator_mi_loss, generator_mi_loss = losses.mi_loss_hinge_dis, losses.mi_loss_hinge_gen
  elif config['MI_loss'] == 'identity':
    discriminator_mi_loss, generator_mi_loss = losses.mi_loss_idt_dis, losses.mi_loss_idt_gen
  else:
    raise NotImplementedError

  def train(x, y):
    G.optim.zero_grad()
    D.optim.zero_grad()
    # How many chunks to split x and y into?
    x = torch.split(x, config['batch_size'])
    y = torch.split(y, config['batch_size'])
    batch_size = config['batch_size']
    counter = 0
    
    # Optionally toggle D and G's "require_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, True)
      utils.toggle_grad(G, False)
      
    for step_index in range(config['num_D_steps']):
      # If accumulating gradients, loop multiple times before an optimizer step
      D.optim.zero_grad()
      for accumulation_index in range(config['num_D_accumulations']):
        z_.sample_()
        y_.sample_()
        gybar = y_[torch.randperm(batch_size), ...] if D.TQ or D.TP else None
        dybar = y[counter][torch.randperm(batch_size), ...] if D.TP or D.TQ else None
        D_fake, D_real, ac, tac, tp, tpbar, tq, tqbar, sx, sp, sq = GD(z_[:config['batch_size']], y_[:config['batch_size']],
                                                           x[counter], y[counter], gybar, dybar,
                                                           train_G=False, split_D=config['split_D'],
                                                           add_bias=True, output_inner_prod=config['use_hybrid'])
        if config['add_log_ratio_y']:
          D_fake, D_real = D_fake + logpy(y_) - logqy(y_), D_real + logpy(y[counter]) - logqy(y[counter])
        # Compute components of D's loss, average them, and divide by 
        # the number of gradient accumulations
        if config['use_hybrid']:
          D_fake = D_fake + ac[:batch_size] - tac[:batch_size]
          D_real = D_real + ac[batch_size:] - tac[batch_size:]
        if config['adaptive_loss']:
          weight_fake = torch.exp(-2.*sx[:batch_size])
          weight_real = torch.exp(-2.*sx[batch_size:])
          if config['use_scaled_bce_logits_with_weighted_ce_loss'] == 'inside':
            D_loss_real, D_loss_fake = discriminator_loss(D_fake * weight_fake, D_real * weight_real)
          else:
            D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real, weight_fake, weight_real)
            D_loss_fake += torch.mean(sx[:batch_size])
            D_loss_real += torch.mean(sx[batch_size:])
        else:
          D_loss_real, D_loss_fake = discriminator_loss(D_fake, D_real)
        MI_P = 0.
        MI_Q = 0.
        # MI on real
        weight_P = torch.exp(-2.*sp[batch_size:]) if config['adaptive_loss'] else 1.
        log_sigma_P = torch.mean(sp[batch_size:]) if config['adaptive_loss'] else 0.
        if config['MI_P_type'] == 'mine':
          tp = tp[batch_size:] - logpy(y[counter])
          tpbar = tpbar[batch_size:] - logpy(dybar)
          tpbar_max = tpbar.max().detach()
          log_sum_exp_tp_ = tpbar_max + torch.log(torch.mean(torch.exp(tpbar - tpbar_max)))
          MI_P = torch.mean(tp) - log_sum_exp_tp_
          MI_P_loss = -MI_P
        elif config['MI_P_type'] == 'eta':
          tp = tp[batch_size:] - logpy(y[counter])
          tpbar = tpbar[batch_size:] - logpy(dybar)
          MI_P = torch.mean(tp) - torch.mean(torch.exp(tpbar))
          MI_P_loss = -MI_P
        elif config['MI_P_type'] == 'ce':
          LL_P = tp[batch_size:]
          MI_P = logpy.entropy + torch.mean(LL_P)
          MI_P_loss = -torch.mean(weight_P * LL_P) + log_sigma_P
        else:
          raise NotImplementedError
        # MI on fake
        weight_Q = torch.exp(-2.*sq[:batch_size]) if config['adaptive_loss'] else 1.
        log_sigma_Q = torch.mean(sq[:batch_size]) if config['adaptive_loss'] else 0.
        if config['MI_Q_type'] == 'mine':
          tq = tq[:batch_size] - logqy(y_)
          tqbar = tqbar[:batch_size] - logqy(gybar)
          tqbar_max = tqbar.max().detach()
          log_sum_exp_tq_ = tqbar_max + torch.log(torch.mean(torch.exp(tqbar - tqbar_max)))
          MI_Q = torch.mean(tq) - log_sum_exp_tq_
          MI_Q_loss = -MI_Q
        elif config['MI_Q_type'] == 'eta':
          tq = tq[:batch_size] - logqy(y_)
          tqbar = tqbar[:batch_size] - logqy(gybar)
          MI_Q = torch.mean(tq) - torch.mean(torch.exp(tqbar))
          MI_Q_loss = -MI_Q
        elif config['MI_Q_type'] == 'ce':
          LL_Q = tq[:batch_size]
          MI_Q = logqy.entropy + torch.mean(LL_Q)
          MI_Q_loss = -torch.mean(weight_Q * LL_Q) + log_sigma_Q
        else:
          raise NotImplementedError
        D_loss = (D_loss_real + D_loss_fake) / float(config['num_D_accumulations'])
        MI_loss = (MI_P_loss + MI_Q_loss) / float(config['num_D_accumulations'])
        D_loss_full = D_loss + MI_loss * config['MI_weight']
        D_loss_full.backward()
        counter += 1
        
      # Optionally apply ortho reg in D
      if config['D_ortho'] > 0.0:
        # Debug print to indicate we're using ortho reg in D.
        print('using modified ortho reg in D')
        utils.ortho(D, config['D_ortho'])
      
      D.optim.step()
    
    # Optionally toggle "requires_grad"
    if config['toggle_grads']:
      utils.toggle_grad(D, False)
      utils.toggle_grad(G, True)
      
    # Zero G's gradients by default before training G, for safety
    G.optim.zero_grad()
    
    # If accumulating gradients, loop multiple times
    for accumulation_index in range(config['num_G_accumulations']):    
      z_.sample_()
      y_.sample_()
      gybar = y_[torch.randperm(batch_size), ...] if D.TQ else None
      D_fake, ac, tac, tp, tpbar, tq, tqbar, sx, sp, sq = GD(z_, y_, gybar=gybar,
                                                 train_G=True, split_D=config['split_D'],
                                                 add_bias=False, output_inner_prod=config['use_hybrid'])
      if config['add_log_ratio_y']:
        D_fake = D_fake + logpy(y_) - logqy(y_)
      if config['use_hybrid']:
        D_fake = D_fake + ac - tac
        MI_loss = 0.
      else:
        # f-divergence (Reverse-KL)
        # minimize log(q) - log(p)
        if config['adaptive_loss']:
          weight_P = torch.exp(-2.*sp)
          weight_Q = torch.exp(-2.*sq)
          if config['adaptive_gen_loss_detach']:
            MI_loss = generator_mi_loss(weight_P.detach()*tp - weight_Q.detach()*tq)
          else:
            MI_loss = generator_mi_loss(weight_P*tp - weight_Q*tq) + torch.mean(sp - sq)
        else:
          MI_loss = generator_mi_loss(tp - tq)
      if config['adaptive_loss']:
        weight_fake = torch.exp(-2.*sx)
        # G_loss is log density ratio, which is not always positive, so penalty is always ignored
        if config['adaptive_gen_loss_detach']:
          G_loss = generator_loss(weight_fake.detach() * D_fake)
        else:
          G_loss = generator_loss(weight_fake * D_fake)
      else:
        G_loss = generator_loss(D_fake)
      G_loss_full = (G_loss + MI_loss * config['MI_weight']) / float(config['num_G_accumulations'])
      G_loss_full.backward()
    
    # Optionally apply modified ortho reg in G
    if config['G_ortho'] > 0.0:
      print('using modified ortho reg in G') # Debug print to indicate we're using ortho reg in G
      # Don't ortho reg shared, it makes no sense. Really we should blacklist any embeddings for this
      utils.ortho(G, config['G_ortho'], 
                  blacklist=[param for param in G.shared.parameters()])
    G.optim.step()
    
    # If we have an ema, update it, regardless of if we test with it or not
    if config['ema']:
      ema.update(state_dict['itr'])
    
    out = {'G_loss': float(G_loss.item()), 
           'D_loss_real': float(D_loss_real.item()),
           'D_loss_fake': float(D_loss_fake.item()),
           'MI_P': float(MI_P.item()),
           'MI_Q': float(MI_Q.item())}
    if not config['use_hybrid']:
      out['MI_loss'] = float(MI_loss.item())
    if config['adaptive_loss']:
      out['weight_fake'] = float(torch.mean(weight_fake).item())
      out['weight_real'] = float(torch.mean(weight_real).item())
      out['weight_P'] = float(torch.mean(weight_P).item())
      out['weight_Q'] = float(torch.mean(weight_Q).item())
    # Return G's loss and the components of D's loss.
    return out
  return train


''' This function takes in the model, saves the weights (multiple copies if 
    requested), and prepares sample sheets: one consisting of samples given
    a fixed noise seed (to show how the model evolves throughout training),
    a set of full conditional sample sheets, and a set of interp sheets.
    This is called during training by save_every.
'''
def save_and_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, 
                    state_dict, config, experiment_name):
  utils.save_weights(G, D, state_dict, config['weights_root'],
                     experiment_name, None, G_ema if config['ema'] else None)
  # Save an additional copy to mitigate accidental corruption if process
  # is killed during a save (it's happened to me before -.-)
  if config['num_save_copies'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'copy%d' %  state_dict['save_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_num'] = (state_dict['save_num'] + 1 ) % config['num_save_copies']
  
  if config['save_test_iteration'] > 0:
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name,
                       'itr%d' %  state_dict['itr'],
                       G_ema if config['ema'] else None)
    
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/fixed_samples%d.jpg' % (config['samples_root'], 
                                                  experiment_name,
                                                  state_dict['itr'])
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                             nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict.get(config['dataset'], config['num_classes_per_sheet']),
                     num_classes=config['n_classes'],
                     samples_per_class=10, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=state_dict['itr'],
                     z_=z_)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=state_dict['itr'],
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This is called in test.
'''
def test_sample(G, D, G_ema, z_, y_, fixed_z, fixed_y, state_dict, config, experiment_name, prefix='test', suffix='',
                sample=None, use_drs=False, drs_classes=[]):
  prefix = prefix+'_' if prefix else ''
  # Use EMA G for samples or non-EMA?
  which_G = G_ema if config['ema'] and config['use_ema'] else G
  
  # Accumulate standing statistics?
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                           z_, y_, config['n_classes'],
                           config['num_standing_accumulations'])
  
  # Save a random sample sheet with fixed z and y      
  with torch.no_grad():
    if config['parallel']:
      fixed_Gz =  nn.parallel.data_parallel(which_G, (fixed_z, which_G.shared(fixed_y)))
    else:
      fixed_Gz = which_G(fixed_z, which_G.shared(fixed_y))
  if not os.path.isdir('%s/%s' % (config['samples_root'], experiment_name)):
    os.mkdir('%s/%s' % (config['samples_root'], experiment_name))
  image_filename = '%s/%s/%sfixed_samples_%s.jpg' % (config['samples_root'], 
                                                     experiment_name,
                                                     prefix,
                                                     suffix)
  torchvision.utils.save_image(fixed_Gz.float().cpu(), image_filename,
                               nrow=int(fixed_Gz.shape[0] **0.5), normalize=True)
  # For now, every time we save, also save sample sheets
  utils.sample_sheet(which_G,
                     classes_per_sheet=utils.classes_per_sheet_dict.get(config['dataset'], config['num_classes_per_sheet']),
                     num_classes=config['n_classes'],
                     samples_per_class=40, parallel=config['parallel'],
                     samples_root=config['samples_root'],
                     experiment_name=experiment_name,
                     folder_number=f'{prefix}{suffix}',
                     z_=z_, sample=sample, use_drs=use_drs, drs_classes=drs_classes)
  # Also save interp sheets
  for fix_z, fix_y in zip([False, False, True], [False, True, False]):
    utils.interp_sheet(which_G,
                       num_per_sheet=16,
                       num_midpoints=8,
                       num_classes=config['n_classes'],
                       parallel=config['parallel'],
                       samples_root=config['samples_root'],
                       experiment_name=experiment_name,
                       folder_number=f'{prefix}{suffix}',
                       sheet_number=0,
                       fix_z=fix_z, fix_y=fix_y, device='cuda')


''' This function runs the inception metrics code, checks if the results
    are an improvement over the previous best (either in IS or FID, 
    user-specified), logs the results, and saves a best_ copy if it's an 
    improvement.
    This is called during training by test_every.
'''
def test(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
         experiment_name, test_log, tb_writer=None, npy_log_dir=None,
         which_metrics=['IS', 'FID', 'IntraFID', 'LPIPS']):
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                    z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
  IS_mean, IS_std, FID, IntraFID, LPIPS = get_inception_metrics(
                                            sample,
                                            config['num_inception_images'],
                                            config['num_intra_inception_images'],
                                            num_splits=10, num_iters=config['torch_fid_num_iters'],
                                            use_torch_intra=config['use_torch_intra'], use_lpips=config['use_lpips'],
                                            which_metrics=which_metrics)
  IntraFID_mean = IntraFID.mean()
  IntraFID_max  = IntraFID.max()
  LPIPS_mean = LPIPS.mean()
  LPIPS_min = LPIPS.min()
  print('Itr %d: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f, ' \
        'mean-FID is %5.4f, max-FID is %5.4f, mean-LPIPS is %5.4f, min-LPIPS is %5.4f' % (state_dict['itr'],
        IS_mean, IS_std, FID, float(IntraFID_mean), float(IntraFID_max), float(LPIPS_mean), float(LPIPS_min)))
  # If improved over previous best metric, save approrpiate copy
  if ((config['which_best'] == 'IS' and IS_mean > state_dict['best_IS'])
    or (config['which_best'] == 'FID' and FID < state_dict['best_FID'])
    or (config['which_best'] == 'meanFID' and IntraFID_mean < state_dict['best_meanFID'])
    or (config['which_best'] == 'maxFID' and IntraFID_mean < state_dict['best_maxFID'])
    or (config['which_best'] == 'meanLPIPS' and LPIPS_mean > state_dict['best_meanLPIPS'])
    or (config['which_best'] == 'minLPIPS' and LPIPS_min > state_dict['best_minLPIPS'])):
    print('%s improved over previous best, saving checkpoint...' % config['which_best'])
    utils.save_weights(G, D, state_dict, config['weights_root'],
                       experiment_name, 'best%d' % state_dict['save_best_num'],
                       G_ema if config['ema'] else None)
    state_dict['save_best_num'] = (state_dict['save_best_num'] + 1 ) % config['num_best_copies']
  state_dict['best_IS'] = max(state_dict['best_IS'], IS_mean)
  state_dict['best_FID'] = min(state_dict['best_FID'], FID)
  state_dict['best_meanFID'] = min(state_dict['best_meanFID'], IntraFID_mean)
  state_dict['best_maxFID'] = min(state_dict['best_maxFID'], IntraFID_max)
  state_dict['best_meanLPIPS'] = max(state_dict['best_meanLPIPS'], LPIPS_mean)
  state_dict['best_minLPIPS'] = max(state_dict['best_minLPIPS'], LPIPS_min)
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), IS_mean=float(IS_mean), IS_std=float(IS_std),
               FID=float(FID), meanFID=IntraFID_mean, maxFID=IntraFID_max,
               meanLPIPS=LPIPS_mean, minLPIPS=LPIPS_min)
  if tb_writer is not None:
    tb_writer.add_scalar('Test/IS_mean', float(IS_mean), int(state_dict['itr']))
    tb_writer.add_scalar('Test/IS_std', float(IS_std), int(state_dict['itr']))
    tb_writer.add_scalar('Test/FID', float(FID), int(state_dict['itr']))
    tb_writer.add_scalar('Test/meanFID', float(IntraFID_mean), int(state_dict['itr']))
    tb_writer.add_scalar('Test/maxFID', float(IntraFID_max), int(state_dict['itr']))
    tb_writer.add_scalar('Test/LPIPS', float(LPIPS_mean), int(state_dict['itr']))
  if npy_log_dir is not None:
    utils.append_npy(os.path.join(npy_log_dir, 'itr.npy'), int(state_dict['itr']))
    utils.append_npy(os.path.join(npy_log_dir, 'IS_mean.npy'), IS_mean)
    utils.append_npy(os.path.join(npy_log_dir, 'IS_std.npy'), IS_std)
    utils.append_npy(os.path.join(npy_log_dir, 'FID.npy'), FID)
    utils.append_npy(os.path.join(npy_log_dir, 'IntraFID.npy'), IntraFID)
    utils.append_npy(os.path.join(npy_log_dir, 'LPIPS.npy'), LPIPS)


''' Test without save.
    This is called in test.
'''
def test_metric(G, D, G_ema, z_, y_, state_dict, config, sample, get_inception_metrics,
                experiment_name, test_log, npy_log_dir=None, prefix='test', suffix='',
                which_metrics=['IS', 'FID', 'IntraFID', 'LPIPS'],
                use_drs=False, drs_classes=[]):
  prefix = prefix+'_' if prefix else ''
  print('Gathering inception metrics...')
  if config['accumulate_stats']:
    utils.accumulate_standing_stats(G_ema if config['ema'] and config['use_ema'] else G,
                                    z_, y_, config['n_classes'],
                                    config['num_standing_accumulations'])
  IS_mean, IS_std, FID, IntraFID, LPIPS = get_inception_metrics(sample, 
                                            config['num_inception_images'],
                                            config['num_intra_inception_images'],
                                            num_splits=10, num_iters=config['torch_fid_num_iters'],
                                            use_torch_intra=config['use_torch_intra'],
                                            use_lpips=config['use_lpips'],
                                            which_metrics=which_metrics,
                                            use_drs=use_drs, drs_classes=drs_classes)
  IntraFID_mean = IntraFID.mean()
  IntraFID_max = IntraFID.max()
  LPIPS_mean = LPIPS.mean()
  LPIPS_min = LPIPS.min()
  print('Itr %d %s: PYTORCH UNOFFICIAL Inception Score is %3.3f +/- %3.3f, PYTORCH UNOFFICIAL FID is %5.4f, ' \
        'mean-FID is %5.4f, max-FID is %5.4f, mean-LPIPS is %5.4f, min-LPIPS is %5.4f' % (state_dict['itr'], suffix,
        IS_mean, IS_std, FID, float(IntraFID_mean), float(IntraFID_max), float(LPIPS_mean), float(LPIPS_min)))
  # Log results to file
  test_log.log(itr=int(state_dict['itr']), suffix=suffix,
               IS_mean=float(IS_mean), IS_std=float(IS_std),
               FID=float(FID), meanFID=IntraFID_mean, maxFID=IntraFID_max,
               meanLPIPS=LPIPS_mean, minLPIPS=LPIPS_min)
  if npy_log_dir is not None:
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}itr.npy'), int(state_dict['itr']))
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}IS_mean.npy'), IS_mean)
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}IS_std.npy'), IS_std)
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}FID.npy'), FID)
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}IntraFID.npy'), IntraFID)
    utils.append_npy(os.path.join(npy_log_dir, f'{prefix}LPIPS.npy'), LPIPS)


def inception_training_function(model, optimizer, state_dict, config):
  criterion = nn.CrossEntropyLoss()

  def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

  def train(x, y):
    optimizer.zero_grad()
    if x.shape[2] != 299 or x.shape[3] != 299:
      x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    outputs, aux_outputs = model(x)
    loss1 = criterion(outputs, y)
    loss2 = criterion(aux_outputs, y)
    loss = loss1 + 0.4*loss2
    loss.backward()
    optimizer.step()
    accuracy = compute_acc(outputs, y)
    return {'loss1': float(loss1.item()),
            'loss2': float(loss2.item()),
            'loss': float(loss.item()),
            'acc': accuracy}
  return train

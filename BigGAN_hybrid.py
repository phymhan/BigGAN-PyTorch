import numpy as np
import math
import functools

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter as P
from torch.nn.utils import spectral_norm

import layers
from utils import make_linear
from sync_batchnorm import SynchronizedBatchNorm2d as SyncBatchNorm2d


# Architectures for G
# Attention is passed in in the format '32_64' to mean applying an attention
# block at both resolution 32x32 and 64x64. Just '64' will apply at 64x64.
def G_arch(ch=64, attention='64', ksize='333333', dilation='111111'):
  arch = {}
  arch[512] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2, 1]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1, 1]],
               'upsample' : [True] * 7,
               'resolution' : [8, 16, 32, 64, 128, 256, 512],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,10)}}
  arch[256] = {'in_channels' :  [ch * item for item in [16, 16, 8, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16,  8, 8, 4, 2, 1]],
               'upsample' : [True] * 6,
               'resolution' : [8, 16, 32, 64, 128, 256],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,9)}}
  arch[128] = {'in_channels' :  [ch * item for item in [16, 16, 8, 4, 2]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2, 1]],
               'upsample' : [True] * 5,
               'resolution' : [8, 16, 32, 64, 128],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,8)}}
  arch[64]  = {'in_channels' :  [ch * item for item in [16, 16, 8, 4]],
               'out_channels' : [ch * item for item in [16, 8, 4, 2]],
               'upsample' : [True] * 4,
               'resolution' : [8, 16, 32, 64],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,7)}}
  arch[32]  = {'in_channels' :  [ch * item for item in [4, 4, 4]],
               'out_channels' : [ch * item for item in [4, 4, 4]],
               'upsample' : [True] * 3,
               'resolution' : [8, 16, 32],
               'attention' : {2**i: (2**i in [int(item) for item in attention.split('_')])
                              for i in range(3,6)}}

  return arch

class Generator(nn.Module):
  def __init__(self, G_ch=64, dim_z=128, bottom_width=4, resolution=128,
               G_kernel_size=3, G_attn='64', n_classes=1000,
               num_G_SVs=1, num_G_SV_itrs=1,
               G_shared=True, shared_dim=0, hier=False,
               cross_replica=False, mybn=False,
               G_activation=nn.ReLU(inplace=False),
               G_lr=5e-5, G_B1=0.0, G_B2=0.999, adam_eps=1e-8,
               BN_eps=1e-5, SN_eps=1e-12, G_mixed_precision=False, G_fp16=False,
               G_init='ortho', skip_init=False, no_optim=False,
               G_param='SN', norm_style='bn', use_torch_SN=False,
               **kwargs):
    super(Generator, self).__init__()
    # Channel width mulitplier
    self.ch = G_ch
    # Dimensionality of the latent space
    self.dim_z = dim_z
    # The initial spatial dimensions
    self.bottom_width = bottom_width
    # Resolution of the output
    self.resolution = resolution
    # Kernel size?
    self.kernel_size = G_kernel_size
    # Attention?
    self.attention = G_attn
    # number of classes, for use in categorical conditional generation
    self.n_classes = n_classes
    # Use shared embeddings?
    self.G_shared = G_shared
    # Dimensionality of the shared embedding? Unused if not using G_shared
    self.shared_dim = shared_dim if shared_dim > 0 else dim_z
    # Hierarchical latent space?
    self.hier = hier
    # Cross replica batchnorm?
    self.cross_replica = cross_replica
    # Use my batchnorm?
    self.mybn = mybn
    # nonlinearity for residual blocks
    self.activation = G_activation
    # Initialization style
    self.init = G_init
    # Parameterization style
    self.G_param = G_param
    # Normalization style
    self.norm_style = norm_style
    # Epsilon for BatchNorm?
    self.BN_eps = BN_eps
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # fp16?
    self.fp16 = G_fp16
    # Architecture dict
    self.arch = G_arch(self.ch, self.attention)[resolution]
    self.use_torch_SN = use_torch_SN

    # If using hierarchical latents, adjust z
    if self.hier:
      # Number of places z slots into
      self.num_slots = len(self.arch['in_channels']) + 1
      self.z_chunk_size = (self.dim_z // self.num_slots)
      # Recalculate latent dimensionality for even splitting into chunks
      self.dim_z = self.z_chunk_size *  self.num_slots
    else:
      self.num_slots = 1
      self.z_chunk_size = 0

    # Which convs, batchnorms, and linear layers to use
    if self.G_param == 'SN':
      if self.use_torch_SN:
        self.which_conv = lambda *a, **kw: spectral_norm(nn.Conv2d(*a, kernel_size=3, padding=1))
        self.which_linear = lambda *a, **kw: spectral_norm(nn.Linear(*a, **kw))
      else:
        self.which_conv = functools.partial(layers.SNConv2d,
                            kernel_size=3, padding=1,
                            num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                            eps=self.SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                            eps=self.SN_eps)
    else:
      self.which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)
      self.which_linear = nn.Linear
      
    # We use a non-spectral-normed embedding here regardless;
    # For some reason applying SN to G's embedding seems to randomly cripple G
    self.which_embedding = nn.Embedding
    bn_linear = (functools.partial(self.which_linear, bias=False) if self.G_shared
                 else self.which_embedding)
    self.which_bn = functools.partial(layers.ccbn,
                          which_linear=bn_linear,
                          cross_replica=self.cross_replica,
                          mybn=self.mybn,
                          input_size=(self.shared_dim + self.z_chunk_size if self.G_shared
                                      else self.n_classes),
                          norm_style=self.norm_style,
                          eps=self.BN_eps)


    # Prepare model
    # If not using shared embeddings, self.shared is just a passthrough
    self.shared = (self.which_embedding(n_classes, self.shared_dim) if G_shared 
                    else layers.identity())
    # First linear layer
    self.linear = self.which_linear(self.dim_z // self.num_slots,
                                    self.arch['in_channels'][0] * (self.bottom_width **2))

    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    # while the inner loop is over a given block
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.GBlock(in_channels=self.arch['in_channels'][index],
                             out_channels=self.arch['out_channels'][index],
                             which_conv=self.which_conv,
                             which_bn=self.which_bn,
                             activation=self.activation,
                             upsample=(functools.partial(F.interpolate, scale_factor=2)
                                       if self.arch['upsample'][index] else None))]]

      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in G at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index], self.which_conv)]

    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

    # output layer: batchnorm-relu-conv.
    # Consider using a non-spectral conv here
    self.output_layer = nn.Sequential(layers.bn(self.arch['out_channels'][-1],
                                                cross_replica=self.cross_replica,
                                                mybn=self.mybn),
                                    self.activation,
                                    self.which_conv(self.arch['out_channels'][-1], 3))

    # Initialize weights. Optionally skip init for testing.
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    # If this is an EMA copy, no need for an optim, so just return now
    if no_optim:
      return
    self.lr, self.B1, self.B2, self.adam_eps = G_lr, G_B1, G_B2, adam_eps
    if G_mixed_precision:
      print('Using fp16 adam in G...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                           betas=(self.B1, self.B2), weight_decay=0,
                           eps=self.adam_eps)

    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d) 
          or isinstance(module, nn.Linear) 
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    print('Param count for G''s initialized parameters: %d' % self.param_count)

  # Note on this forward function: we pass in a y vector which has
  # already been passed through G.shared to enable easy class-wise
  # interpolation later. If we passed in the one-hot and then ran it through
  # G.shared in this forward function, it would be harder to handle.
  def forward(self, z, y):
    # If hierarchical, concatenate zs and ys
    if self.hier:
      zs = torch.split(z, self.z_chunk_size, 1)
      z = zs[0]
      ys = [torch.cat([y, item], 1) for item in zs[1:]]
    else:
      ys = [y] * len(self.blocks)
      
    # First linear layer
    h = self.linear(z)
    # Reshape
    h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
    
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      # Second inner loop in case block has multiple layers
      for block in blocklist:
        h = block(h, ys[index])
        
    # Apply batchnorm-relu-conv-tanh at output
    return torch.tanh(self.output_layer(h))


# Discriminator architecture, same paradigm as G's above
def D_arch(ch=64, attention='64',ksize='333333', dilation='111111'):
  arch = {}
  arch[256] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 8, 16, 16]],
               'downsample' : [True] * 6 + [False],
               'resolution' : [128, 64, 32, 16, 8, 4, 4 ],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[128] = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8, 16]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16, 16]],
               'downsample' : [True] * 5 + [False],
               'resolution' : [64, 32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,8)}}
  arch[64]  = {'in_channels' :  [3] + [ch*item for item in [1, 2, 4, 8]],
               'out_channels' : [item * ch for item in [1, 2, 4, 8, 16]],
               'downsample' : [True] * 4 + [False],
               'resolution' : [32, 16, 8, 4, 4],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,7)}}
  arch[32]  = {'in_channels' :  [3] + [item * ch for item in [4, 4, 4]],
               'out_channels' : [item * ch for item in [4, 4, 4, 4]],
               'downsample' : [True, True, False, False],
               'resolution' : [16, 16, 16, 16],
               'attention' : {2**i: 2**i in [int(item) for item in attention.split('_')]
                              for i in range(2,6)}}
  return arch

class Discriminator(nn.Module):

  def __init__(self, D_ch=64, D_wide=True, resolution=128,
               D_kernel_size=3, D_attn='64', n_classes=1000,
               num_D_SVs=1, num_D_SV_itrs=1, D_activation=nn.ReLU(inplace=False),
               D_lr=2e-4, D_B1=0.0, D_B2=0.999, adam_eps=1e-8,
               SN_eps=1e-12, output_dim=1, D_mixed_precision=False, D_fp16=False,
               D_init='ortho', skip_init=False, D_param='SN',
               projection=True, AC=False, TAC=False, dis_fc_dim=[1], use_torch_SN=False,
               adaptive_loss='none', adaptive_loss_param='none', detach_weight_linear=False,
               over_parameterize=False, naive_hybrid=False, no_proj_bias=False,
               **kwargs):
    super(Discriminator, self).__init__()
    # Width multiplier
    self.ch = D_ch
    # Use Wide D as in BigGAN and SA-GAN or skinny D as in SN-GAN?
    self.D_wide = D_wide
    # Resolution
    self.resolution = resolution
    # Kernel size
    self.kernel_size = D_kernel_size
    # Attention?
    self.attention = D_attn
    # Number of classes
    self.n_classes = n_classes
    # Activation
    self.activation = D_activation
    # Initialization style
    self.init = D_init
    # Parameterization style
    self.D_param = D_param
    # Epsilon for Spectral Norm?
    self.SN_eps = SN_eps
    # Fp16?
    self.fp16 = D_fp16
    # Architecture
    self.arch = D_arch(self.ch, self.attention)[resolution]
    # Additional
    self.projection = projection
    self.AC = AC
    self.TAC = TAC
    self.use_torch_SN = use_torch_SN
    self.adaptive_loss = adaptive_loss
    self.adaptive_loss_param = adaptive_loss_param
    self.detach_weight_linear = detach_weight_linear
    self.over_parameterize = over_parameterize
    self.scalar_sigmoid = nn.Parameter(torch.tensor(0.))
    self.scalar_sigma_x = nn.Parameter(torch.tensor(0.))
    self.scalar_sigma_y = nn.Parameter(torch.tensor(0.))
    self.naive_hybrid = naive_hybrid
    if self.naive_hybrid:
      self.projection = self.AC = self.TAC = False

    # Which convs, batchnorms, and linear layers to use
    # No option to turn off SN in D right now
    if self.D_param == 'SN':
      if self.use_torch_SN:
        self.which_conv = lambda *a, **kw: spectral_norm(nn.Conv2d(*a, kernel_size=3, padding=1))
        self.which_linear = lambda *a, **kw: spectral_norm(nn.Linear(*a, **kw))
        self.which_embedding = lambda *a, **kw: spectral_norm(nn.Embedding(*a, **kw))
      else:
        self.which_conv = functools.partial(layers.SNConv2d,
                            kernel_size=3, padding=1,
                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                            eps=self.SN_eps)
        self.which_linear = functools.partial(layers.SNLinear,
                            num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                            eps=self.SN_eps)
        self.which_embedding = functools.partial(layers.SNEmbedding,
                                num_svs=num_D_SVs, num_itrs=num_D_SV_itrs,
                                eps=self.SN_eps)
    # Prepare model
    # self.blocks is a doubly-nested list of modules, the outer loop intended
    # to be over blocks at a given resolution (resblocks and/or self-attention)
    self.blocks = []
    for index in range(len(self.arch['out_channels'])):
      self.blocks += [[layers.DBlock(in_channels=self.arch['in_channels'][index],
                       out_channels=self.arch['out_channels'][index],
                       which_conv=self.which_conv,
                       wide=self.D_wide,
                       activation=self.activation,
                       preactivation=(index > 0),
                       downsample=(nn.AvgPool2d(2) if self.arch['downsample'][index] else None))]]
      # If attention on this block, attach it to the end
      if self.arch['attention'][self.arch['resolution'][index]]:
        print('Adding attention layer in D at resolution %d' % self.arch['resolution'][index])
        self.blocks[-1] += [layers.Attention(self.arch['out_channels'][index],
                                             self.which_conv)]
    # Turn self.blocks into a ModuleList so that it's all properly registered.
    self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
    # Linear output layer. The output dimension is typically 1, but may be
    # larger if we're e.g. turning this into a VAE with an inference output
    assert not dis_fc_dim or dis_fc_dim[-1] == output_dim
    self.linear = make_linear(self.which_linear, self.arch['out_channels'][-1], dis_fc_dim)
    # Embedding for projection discrimination
    if self.projection:
      self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
      if self.over_parameterize:
        self.embedq = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
    if self.AC:
      self.linear_ac = self.which_linear(self.arch['out_channels'][-1], self.n_classes, bias=not no_proj_bias)
    if self.TAC:
      self.linear_tac = self.which_linear(self.arch['out_channels'][-1], self.n_classes, bias=not no_proj_bias)
    # for sigma  : D_real*wx_real, D_fake*wx_fake, AC_real*wp_real, TAC_fake*wq_fake
    # for sigmoid: loss(D_real)*w_real[0], loss(AC_real)*w_real[1], loss(D_fake)*w_fake[0], loss(TAC_fake)*w_fake[1]
    if self.adaptive_loss == 'sigma':
      which_linear = nn.Linear if self.adaptive_loss_param == 'none' else self.which_linear
      self.linear_wx = make_linear(which_linear, self.arch['out_channels'][-1], [1], nl='softplus')
      self.linear_wp = make_linear(which_linear, self.arch['out_channels'][-1], [1], nl='softplus')
      self.linear_wq = make_linear(which_linear, self.arch['out_channels'][-1], [1], nl='softplus')
    if self.adaptive_loss == 'sigmoid':
      which_linear = nn.Linear if self.adaptive_loss_param == 'none' else self.which_linear
      self.linear_wx = make_linear(which_linear, self.arch['out_channels'][-1], [1], nl='none')
    
    if self.naive_hybrid:
      self.embed = self.which_embedding(self.n_classes, self.arch['out_channels'][-1])
      self.psi = self.which_linear(self.arch['out_channels'][-1], 1)
      self.linear_ac = self.which_linear(self.arch['out_channels'][-1], self.n_classes, bias=not no_proj_bias)
      self.linear_tac = self.which_linear(self.arch['out_channels'][-1], self.n_classes, bias=not no_proj_bias)

    # Initialize weights
    if not skip_init:
      self.init_weights()

    # Set up optimizer
    self.lr, self.B1, self.B2, self.adam_eps = D_lr, D_B1, D_B2, adam_eps
    if D_mixed_precision:
      print('Using fp16 adam in D...')
      import utils
      self.optim = utils.Adam16(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    else:
      self.optim = optim.Adam(params=self.parameters(), lr=self.lr,
                             betas=(self.B1, self.B2), weight_decay=0, eps=self.adam_eps)
    # LR scheduling, left here for forward compatibility
    # self.lr_sched = {'itr' : 0}# if self.progressive else {}
    # self.j = 0

  # Initialize
  def init_weights(self):
    self.param_count = 0
    for module in self.modules():
      if (isinstance(module, nn.Conv2d)
          or isinstance(module, nn.Linear)
          or isinstance(module, nn.Embedding)):
        if self.init == 'ortho':
          init.orthogonal_(module.weight)
        elif self.init == 'N02':
          init.normal_(module.weight, 0, 0.02)
        elif self.init in ['glorot', 'xavier']:
          init.xavier_uniform_(module.weight)
        else:
          print('Init style not recognized...')
        self.param_count += sum([p.data.nelement() for p in module.parameters()])
    # init as zero
    if self.adaptive_loss_param == 'none':
      for name in ['linear_wx', 'linear_wp', 'linear_wq']:
        param = getattr(self, name, None)
        if isinstance(param, nn.Sequential):
          param = param[-2]  # the last fc layer is identity by default
        w = getattr(param, 'weight', None)
        b = getattr(param, 'bias', None)
        if w is not None:
          w.data.fill_(0.)
        if b is not None:
          b.data.fill_(0.)
          if self.adaptive_loss == 'sigma':
            b.data.fill_(-5.)  # log(wt) is more close to zero
    print('Param count for D''s initialized parameters: %d' % self.param_count)

  def forward(self, x, y=None):
    # Stick x into h for cleaner for loops without flow control
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])  # h is phi(x)
    # Get initial class-unconditional output
    out = self.linear(h) if self.linear is not None else 0.
    out_ac = None
    out_tac = None
    out_wx = None
    out_wp = None
    out_wq = None

    # Get projection of final featureset onto class vectors and add to evidence
    if self.projection:
      out = out + torch.sum(self.embed(y) * h, 1, keepdim=True)
      if self.over_parameterize:
        out = out - torch.sum(self.embedq(y) * h, 1, keepdim=True)
    if self.AC:
      out_ac = self.linear_ac(h)
    if self.TAC:
      out_tac = self.linear_tac(h)
    if self.adaptive_loss == 'sigma':
      out_wx = self.linear_wx(h.detach()) if self.detach_weight_linear else self.linear_wx(h)
      out_wp = self.linear_wp(h.detach()) if self.detach_weight_linear else self.linear_wp(h)
      out_wq = self.linear_wq(h.detach()) if self.detach_weight_linear else self.linear_wq(h)
      out_wx = torch.squeeze(out_wx)
      out_wp = torch.squeeze(out_wp)
      out_wq = torch.squeeze(out_wq)
    if self.adaptive_loss == 'sigmoid':
      out_wx = self.linear_wx(h.detach()) if self.detach_weight_linear else self.linear_wx(h)
      out_wx = torch.squeeze(out_wx)
    
    if self.naive_hybrid:
      out_ac = self.linear_ac(h)
      out_tac = self.linear_tac(h)
      out_pd = self.psi(h) + torch.sum(self.embed(y) * h, 1, keepdim=True)
      return out.squeeze(1), out_ac, out_tac, out_pd, None, None

    return out.squeeze(1), out_ac, out_tac, out_wx, out_wp, out_wq
  
  def get_embedding(self, x, y=None):
    # Get data embedding
    h = x
    # Loop over blocks
    for index, blocklist in enumerate(self.blocks):
      for block in blocklist:
        h = block(h)
    # Apply global sum pooling as in SN-GAN
    h = torch.sum(self.activation(h), [2, 3])  # h is phi(x)
    out = h
    return out


# Parallelized G_D to minimize cross-gpu communication
# Without this, Generator outputs would get all-gathered and then rebroadcast.
class G_D(nn.Module):
  def __init__(self, G, D, **kwargs):
    super(G_D, self).__init__()
    self.G = G
    self.D = D

  def forward(self, z, gy, x=None, dy=None, train_G=False, return_G_z=False,
              split_D=False):              
    # If training G, enable grad tape
    with torch.set_grad_enabled(train_G):
      # Get Generator output given noise
      G_z = self.G(z, self.G.shared(gy))
      # Cast as necessary
      if self.G.fp16 and not self.D.fp16:
        G_z = G_z.float()
      if self.D.fp16 and not self.G.fp16:
        G_z = G_z.half()
    # Split_D means to run D once with real data and once with fake,
    # rather than concatenating along the batch dimension.
    if split_D:
      D_fake, ac_fake, tac_f, wx_f, wp_f, wq_f, w_f = self.D(G_z, gy)
      if x is not None:
        D_real, ac_real, tac_real, wx_r, wp_r, wq_r, w_r = self.D(x, dy)
        ac = torch.cat([ac_fake, ac_real], 0) if ac_fake is not None else None
        tac = torch.cat([tac_fake, tac_real], 0) if tac_fake is not None else None
        wx = torch.cat([wx_f, wx_r], 0) if wx_f is not None else None
        wp = torch.cat([wp_f, wp_r], 0) if wp_f is not None else None
        wq = torch.cat([wq_f, wq_r], 0) if wq_f is not None else None
        return D_fake, D_real, ac, tac, wx, wp, wq
      else:
        if return_G_z:
          return D_fake, G_z, ac_fake, tac_fake, wx_f, wp_f, wq_f
        else:
          return D_fake, ac_fake, tac_fake, wx_f, wp_f, wq_f
    # If real data is provided, concatenate it with the Generator's output
    # along the batch dimension for improved efficiency.
    else:
      D_input = torch.cat([G_z, x], 0) if x is not None else G_z
      D_class = torch.cat([gy, dy], 0) if dy is not None else gy
      # Get Discriminator output
      D_out, ac, tac, wx, wp, wq = self.D(D_input, D_class)
      if x is not None:
        D_fake, D_real = torch.split(D_out, [G_z.shape[0], x.shape[0]])
        return D_fake, D_real, ac, tac, wx, wp, wq
      else:
        if return_G_z:
          return D_out, G_z, ac, tac, wx, wp, wq
        else:
          return D_out, ac, tac, wx, wp, wq

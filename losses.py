import torch
import torch.nn.functional as F

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real, weight_fake=None, weight_real=None):
  if weight_real is None:
    loss_real = torch.mean(F.relu(1. - dis_real))
  else:
    loss_real = torch.mean(weight_real.reshape(dis_real.size()) * F.relu(1. - dis_real))
  if weight_fake is None:
    loss_fake = torch.mean(F.relu(1. + dis_fake))
  else:
    loss_fake = torch.mean(weight_fake.reshape(dis_fake.size()) * F.relu(1. + dis_fake))
  return loss_real, loss_fake


def loss_hinge_gen(dis_fake, weight_fake=None):
  if weight_fake is not None:
    weight_fake = weight_fake.reshape(dis_fake.size())
  if weight_fake is None:
    loss = -torch.mean(dis_fake)
  else:
    loss = -torch.mean(weight_fake * dis_fake)
  return loss


# Vanilla
def loss_bce_dis(dis_fake, dis_real, weight_fake=None, weight_real=None):
  if weight_fake is not None:
    weight_fake = weight_fake.reshape(dis_fake.size())
  if weight_real is not None:
    weight_real = weight_real.reshape(dis_real.size())
  # target_fake = torch.tensor(0.).cuda().expand_as(dis_fake)
  # target_real = torch.tensor(1.).cuda().expand_as(dis_real)
  target_fake = torch.zeros_like(dis_fake)
  target_real = torch.ones_like(dis_real)
  loss_fake = F.binary_cross_entropy_with_logits(dis_fake, target_fake, weight=weight_fake)
  loss_real = F.binary_cross_entropy_with_logits(dis_real, target_real, weight=weight_real)
  return loss_real, loss_fake


def loss_bce_gen(dis_fake, weight_fake=None):
  if weight_fake is not None:
    weight_fake = weight_fake.reshape(dis_fake.size())
  target_real = torch.ones_like(dis_fake)
  # target_real = torch.tensor(1.).cuda().expand_as(dis_fake)
  loss = F.binary_cross_entropy_with_logits(dis_fake, target_real, weight=weight_fake)
  return loss


# LSGAN
def loss_lsgan_dis(dis_fake, dis_real):
  target_fake = torch.tensor(0.).cuda().expand_as(dis_fake)
  target_real = torch.tensor(1.).cuda().expand_as(dis_real)
  loss_fake = F.mse_loss(dis_fake, target_fake)
  loss_real = F.mse_loss(dis_real, target_real)
  return loss_real, loss_fake


def loss_lsgan_gen(dis_fake):
  target_real = torch.tensor(1.).cuda().expand_as(dis_fake)
  loss = F.mse_loss(dis_fake, target_real)
  return loss


# MINE losses
# Hinge Loss
def mi_loss_hinge_dis(output):
  loss = torch.mean(F.relu(1. - output))
  return loss


def mi_loss_hinge_gen(output):
  loss = -torch.mean(output)
  return loss


# Identity
def mi_loss_idt_dis(output):
  loss = -torch.mean(output)
  return loss


def mi_loss_idt_gen(output):
  loss = -torch.mean(output)
  return loss


# f-div losses
# rev-kl
def f_div_loss_revkl(logit_p, logit_q, y, weight=None):
  # loss = F.cross_entropy(logit_p, y) - F.cross_entropy(logit_q, y)
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  if weight is None:
    loss = torch.mean(nllp - nllq)
  else:
    loss = torch.mean(weight.view(-1) * (nllp - nllq))
  return loss


# hinge -1
def f_div_loss_hinge1(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  loss = torch.mean(F.relu(1. + nllp - nllq))
  return loss


# hinge 0
def f_div_loss_hinge0(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  loss = torch.mean(F.relu(nllp - nllq))
  return loss


# revkl but ignore LogSumExp
def f_div_loss_proj(logit_p, logit_q, y):
  logit_p = logit_p[range(y.size(0)), y]
  logit_q = logit_q[range(y.size(0)), y]
  loss = -torch.mean(logit_p-logit_q)
  return loss


# kl
def f_div_loss_kl(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  logr = nllq - nllp
  r = torch.exp(logr)
  loss = torch.mean(r * logr)
  return loss
  

# pearson
def f_div_loss_pearson(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  logr = nllq - nllp
  r = torch.exp(logr)
  loss = torch.mean((r - 1.).pow(2))
  return loss


# squared
def f_div_loss_squared(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  logr = nllq - nllp
  rsq = torch.exp(logr / 2.)
  loss = torch.mean((rsq - 1.).pow(2))
  return loss


def f_div_loss_jsd(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  logr = nllq - nllp
  r = torch.exp(logr)
  loss = torch.mean(-(r + 1.0) * torch.log(0.5 * r + 0.5) + r * logr)
  return loss


def f_div_loss_gan(logit_p, logit_q, y):
  nllp = F.cross_entropy(logit_p, y, reduction='none')
  nllq = F.cross_entropy(logit_q, y, reduction='none')
  logr = nllq - nllp
  r = torch.exp(logr)
  loss = torch.mean(r * logr - (r + 1.0) * torch.log(r + 1.0))
  return loss


# Default to hinge loss
generator_loss = loss_hinge_gen
discriminator_loss = loss_hinge_dis
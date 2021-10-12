import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from argparse import ArgumentParser
import os
import sys
import random
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# sys.path.append('..')
import datasets as dset
# from utils import str2list

def prepare_parser():
  usage = 'Generate image filename index for custom datasets.'
  parser = ArgumentParser(description=usage)
  parser.add_argument('--use_custom_dataset', action='store_true')
  parser.add_argument('--dataset', type=str, default='celeba_gender')
  parser.add_argument('--index_filename', type=str, default='')
  parser.add_argument('--num_classes', type=int, default=2, help='only valid if use_custom_dataset')
  parser.add_argument('--data_root', type=str, default='./data/celeba', help='Default location where data is stored (default: %(default)s)')
  parser.add_argument('--out_root', type=str, default='./')
  parser.add_argument('--limit_batches', type=float, nargs='*', default=[])
  return parser

def run(config):
  # Get loader
  config['drop_last'] = False
  dataset = config['dataset']
  data_root = config['data_root']
  if len(config['limit_batches']) == 0:
    config['limit_batches'] = [1]*config['num_classes']
  index_filename = config['index_filename'] or os.path.join(config['out_root'], '%s_imgs.npz' % dataset)
  if dataset in ['celeba_gender']:
    parition_dict = {}
    with open(os.path.join(data_root, 'list_eval_partition.txt'), 'r') as f:
      for l in f.readlines():
        parition_dict[l.split()[0]] = int(l.split()[1])
    with open(os.path.join(data_root, 'list_attr_celeba.txt'), 'r') as f:
      attr_list = f.readlines()
    attr_map = {n: i for i, n in enumerate(attr_list[1].strip().split())}
    label_idx = attr_map['Male']  # '1' or '-1'
    imgs_dict = {0: [], 1: [], 2: []}
    for l in attr_list[2:]:
      path = os.path.join(data_root, 'img_align_celeba', l.split()[0])
      target = 1 if int(l.split()[label_idx+1]) > 0 else 0
      if parition_dict[l.split()[0]] == 0 and config['limit_batches'][target] < 1:
        if random.random() > config['limit_batches'][target]:
          continue
      imgs_dict[parition_dict[l.split()[0]]].append((path, target))
    np.savez_compressed(index_filename, **{'imgs': imgs_dict[0], 'imgs_val': imgs_dict[1], 'imgs_test': imgs_dict[2]})
  if dataset in ['dog']:
    # Get map_clsloc.txt from https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    with open('map_clsloc.txt', 'r') as f:
      cls2name = {l.strip().split()[0]: l.strip().split()[2] for l in f.readlines()}
    print(cls2name)
    name_keep = [cls2name[c] for c in cls2name.keys() if 'dog' in cls2name[c]]
    print(name_keep)
    name_keep = name_keep[:-1]
    cls_keep = [c for c in cls2name.keys() if cls2name[c] in name_keep]
    # classes, cls2idx = dset.find_classes(data_root)
    cls2idx = {cls_keep[i]: i for i in range(len(cls_keep))}
    imgs = dset.make_subset(data_root, cls_keep, cls2idx)
    np.savez_compressed(index_filename, imgs=imgs)
  if dataset in ['cat']:
    # Get map_clsloc.txt from https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
    with open('map_clsloc.txt', 'r') as f:
      cls2name = {l.strip().split()[0]: l.strip().split()[2] for l in f.readlines()}
    print(cls2name)
    name_keep = [cls2name[c] for c in cls2name.keys() if 'cat' in cls2name[c]]
    print(name_keep)
    name_keep = name_keep[:-2]
    cls_keep = [c for c in cls2name.keys() if cls2name[c] in name_keep]
    # classes, cls2idx = dset.find_classes(data_root)
    cls2idx = {cls_keep[i]: i for i in range(len(cls_keep))}
    imgs = dset.make_subset(data_root, cls_keep, cls2idx)
    np.savez_compressed(index_filename, imgs=imgs)


def main():
  random.seed(42)
  # parse command line
  parser = prepare_parser()
  config = vars(parser.parse_args())
  print(config)
  run(config)


if __name__ == '__main__':    
    main()
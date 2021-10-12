import argparse
import os
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shutil
import random


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def main(config):
    dataset = 'V%d' % config['num_classes']
    if not os.path.exists(os.path.join(config['data_root'], dataset)):
        os.makedirs(os.path.join(config['data_root'], dataset))
    identities = os.listdir(config['src_root'])
    random.seed(0)
    random.shuffle(identities)
    for id in identities[:config['num_classes']]:
        src_dir = os.path.join(config['src_root'], id)
        dst_dir = os.path.join(config['data_root'], dataset, id)
        shutil.copytree(src_dir, dst_dir)
        print(f'id {id} copied.')


parser = argparse.ArgumentParser()
parser.add_argument('--num_classes', type=int, default=200, help='number different identities')
parser.add_argument('--src_root', type=str, default='../data/VGGFace2/train')
parser.add_argument('--data_root', type=str, default='../data')
config = vars(parser.parse_args())
main(config)

import argparse
import os
import torchvision.datasets as dset
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import shutil
import pdb


torch.manual_seed(0)
np.random.seed(0)


def save_as_image_folder():
    dataset = dset.CIFAR100(root='../data/cifar100', download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    dstroot = '../data/cifar100-imagefolder'
    if not os.path.exists(dstroot):
        os.makedirs(dstroot)
    for y in range(100):
        classname = f'{y:03d}'
        if not os.path.exists(os.path.join(dstroot, classname)):
            os.makedirs(os.path.join(dstroot, classname))

    for id, data in enumerate(dataloader, 0):
        x, y = data
        classname = f'{y.item():03d}'
        imagename = f'{id:05d}.png'
        im = Image.fromarray((x * 255).squeeze().permute(1, 2, 0).numpy().astype(np.uint8))
        im.save(os.path.join(dstroot, classname, imagename))
        if id % 1000 == 0:
            print(f'image {classname}/{imagename} saved.')


def make_imbalanced_dataset():
    srcroot = '../data/cifar100-imagefolder'
    dstroot = '../data/cifar100-imbalanced'
    if not os.path.exists(dstroot):
        os.makedirs(dstroot)
    for y in range(100):
        classname = f'{y:03d}'
        if not os.path.exists(os.path.join(dstroot, classname)):
            os.makedirs(os.path.join(dstroot, classname))
    sample_nums = {i: int(n) for i, n in enumerate(np.linspace(500, 100, 100), 0)}
    for y in range(100):
        classname = f'{y:03d}'
        images = os.listdir(os.path.join(srcroot, classname))
        for imagename in images[:sample_nums[y]]:
            shutil.copyfile(os.path.join(srcroot, classname, imagename),
                            os.path.join(dstroot, classname, imagename))
        print(f'class {y} finished.')


if not os.path.exists('../data/cifar100-imagefolder'):
    save_as_image_folder()
if not os.path.exists('../data/cifar100-imbalanced'):
    make_imbalanced_dataset()

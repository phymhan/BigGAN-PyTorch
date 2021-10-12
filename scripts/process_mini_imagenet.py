import os
import shutil
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_root", type=str, default='../../mini-imagenet-tools/mini_imagenet_split/Vinyals/')
    parser.add_argument("--src_root", type=str, default='../data/ImageNet')
    parser.add_argument("--dst_root", type=str, default='../data/Mini-ImageNet-100')
    args = parser.parse_args()

    img_list = []
    for filename in ['trainval.csv', 'test.csv']:
        with open(os.path.join(args.csv_root, filename), 'r') as f:
            img_list += [l.rstrip() for l in f.readlines() if l.startswith('n')]
    
    for img in tqdm(img_list):
        img_name, img_class = img.split(',')
        if not os.path.exists(os.path.join(args.dst_root, img_class)):
            os.makedirs(os.path.join(args.dst_root, img_class))
        shutil.copyfile(
            os.path.join(args.src_root, img_class, img_name),
            os.path.join(args.dst_root, img_class, img_name)
        )

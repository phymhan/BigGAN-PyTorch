import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='sep')
parser.add_argument('--outpath', type=str, default='../figs')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/ACGAN-PyTorch/results_visualize/cifar10/'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/ACGAN-PyTorch/results_visualize/cifar10/'

if config.config == 'sep':
    models = [
        'cgan+ac_detach',
        'tac+naof',
        'fcgan',
    ]
    legend = [
        'projection',
        'ce+ce',
        'mine+mine',
    ]
    epochs = [0, 99, 199]
    classnames = ['airplance', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    exp = config.config
else:
    raise NotImplementedError

for m, l in zip(models, legend):
    for e in epochs:
        fp = os.path.join(log_root, m, 'features')
        f = []
        y = []
        for b in range(4):
            f.append(np.load(os.path.join(fp, f'real_epoch_{e}_batch_{b}_f.npy')))
            f.append(np.load(os.path.join(fp, f'fake_epoch_{e}_batch_{b}_f.npy')))
            y.append(np.load(os.path.join(fp, f'real_epoch_{e}_batch_{b}_y.npy')))
            y.append(np.load(os.path.join(fp, f'fake_epoch_{e}_batch_{b}_y.npy')))
        f = np.concatenate(f, 0)
        y = np.concatenate(y, 0)
        x = TSNE(n_components=2).fit_transform(f)
        df = pd.DataFrame()
        df['t-SNE one'] = x[:,0]
        df['t-SNE two'] = x[:,1]
        df['class'] = [classnames[y_] for y_ in y] if classnames else y
        sns.set()
        sns.set_context('paper')
        fig = plt.figure(figsize=(5,5))
        ax = sns.scatterplot(x='t-SNE one', y='t-SNE two', hue='class', palette=sns.hls_palette(10, h=0.5), data=df, alpha=0.85)
        ax.axis('equal')
        plt.tight_layout()
        fig.savefig(os.path.join(config.outpath, f'{exp}_{l}_ep{e}_rf.pdf'))
        # plt.show()
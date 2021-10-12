import os
import numpy as np
import copy
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pdb


def parse_(s):
    mean, std = s.strip().split(',')
    return float(mean), float(std)


def replicate_std(idx, mean, std):
    idx_ = np.concatenate((idx, idx, idx), 0)
    mean_ = np.concatenate((mean-std, mean, mean+std), 0)
    return idx_, mean_


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='mog')
parser.add_argument('--outpath', type=str, default='../figs')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'

methods = ['fc', 'tac', 'ac', 'hybrid', 'projection']
methods_valid = ['fc', 'tac', 'hybrid', 'projection']
# legend = ['f-cGAN', 'TAC-GAN', 'AC-GAN', 'hybrid-cGAN', 'Projection cGAN']
legend_valid = ['f-cGAN', 'TAC-GAN', 'P2GAN', 'Proj-GAN']
legend_order = ['Proj-GAN', 'TAC-GAN', 'f-cGAN', 'P2GAN']
classes = ['0', '1', '2', 'm']
class_legend = ['Class_0', 'Class_1', 'Class_2', 'Marginal']
distances = [d/2+1 for d in [0, 2, 4, 6, 8]]
# distances = [d/2+1 for d in [4]]

for dist in distances:
    sns.set()
    sns.set_context('paper')
    data = pd.DataFrame()
    for loss in ['bce', 'hinge']:
        for m, l in zip(methods_valid, legend_valid):
            run = np.random.randint(0, 100, 1)[0]
            name = f'{dist:.1f}_{loss}_{run}'
            density = np.load(os.path.join(log_root, name, f'{m}_data.npy'))
            # for c in [0, 1, 2]:
            #     density_ = density[c*10000:(c+1)*10000]
            #     df = pd.DataFrame()
            #     df['density'] = density_
            #     df['method'] = l
            #     df['loss'] = loss
            #     df['class'] = f'class {c}'
            #     if data.empty:
            #         data = df
            #     else:
            #         data = data.append(df)
            # df = pd.DataFrame()
            # df['density'] = density
            # df['method'] = l
            # df['loss'] = loss
            # df['class'] = 'marginal'
            # data = data.append(df)
            for c in [0, 1, 2]:
                density_ = density[c*10000:(c+1)*10000]
                df = pd.DataFrame()
                df[f'class {c}'] = density_
                df['method'] = l
                df['loss'] = loss
                if data.empty:
                    data = df
                else:
                    data = data.append(df)
            df = pd.DataFrame()
            df['marginal'] = density
            df['method'] = l
            df['loss'] = loss
            data = data.append(df)

    df = pd.DataFrame()
    df['class'] = float('nan')
    df['method'] = l
    df['loss'] = loss
    data = data.append(df)
    
    # pdb.set_trace()
    g = sns.FacetGrid(data, row='loss', col='method', sharey=True, margin_titles=False, col_order=legend_order)
    g.map(sns.kdeplot, 'marginal', **{'bw': 0.5, 'color': 'y', 'shade': True, 'legend': True})
    g.map(sns.kdeplot, 'class 0', **{'bw': 0.5, 'color': 'b'})
    g.map(sns.kdeplot, 'class 1', **{'bw': 0.5, 'color': 'r'})
    g.map(sns.kdeplot, 'class 2', **{'bw': 0.5, 'color': 'g'})
    g.map(sns.kdeplot, 'class', **{'bw': 0.5, 'color': 'g'})
    g.add_legend()
    fig = plt.gcf()
    fig.subplots_adjust(right=0.92)
    # plt.show()
    # fig.subplots_adjust(right=0.88)
    fig.savefig(os.path.join(config.outpath, f'mog_d{dist}_plot_iccv.pdf'))

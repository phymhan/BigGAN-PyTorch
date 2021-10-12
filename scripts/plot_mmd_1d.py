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

def fix_file(lines):
    if len(lines) == 25:
        return lines
    if lines[len(lines)-25].startswith('fc:'):
        # pdb.set_trace()
        return lines[len(lines)-25:]
    else:
        pdb.set_trace()
        raise RuntimeError

parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='mog')
parser.add_argument('--outpath', type=str, default='../figs_iccv')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'
else:
    # log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/Dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'

methods = ['fc', 'tac', 'ac', 'hybrid', 'projection']
methods_valid = ['projection', 'tac', 'fc', 'hybrid']
legend_valid = ['Proj-GAN', 'TAC-GAN', 'f-cGAN', 'P2GAN']
classes = ['0', '1', '2', 'm']
class_legend = ['Class_0', 'Class_1', 'Class_2', 'Marginal']
distances = [d/2+1 for d in [0, 2, 4, 6, 8]]
sns.set()
sns.set_context('paper')
data = pd.DataFrame()
for loss in ['bce', 'hinge']:
    res_full = {}
    for dist in distances:
        res_full[dist] = {m: {'0': [], '1': [], '2': [], 'm': []} for m in methods}
        print(f'experiment {loss} {int(dist)}')
        for run in range(100):
            name = f'{dist:.1f}_{loss}_{run}'
            fp = os.path.join(log_root, name, 'results.txt')
            if os.path.exists(fp):
                with open(fp, 'r') as f:
                    lines = f.readlines()
                lines = fix_file(lines)
                for m in methods:
                    for c in classes:
                        mean, std = parse_(lines[methods.index(m)*5+classes.index(c)+1])
                        res_full[dist][m][c].append(mean)
                        # res_full[dist][m][c].append(mean-std)
                        # res_full[dist][m][c].append(mean+std)
            else:
                pdb.set_trace()
                raise RuntimeError
    for c, cl in zip(classes, class_legend):
        for m, l in zip(methods_valid, legend_valid):
            mmd_mean = []
            mmd_std = []
            for dist in distances:
                a = np.array(res_full[dist][m][c])
                a.sort()
                a = a[:int(0.9*len(a))]
                mmd_mean.append(a.mean())
                mmd_std.append(np.sqrt(a.std()))
                # mmd_std.append(a.std())
            distances_, mmd_ = replicate_std(np.array(distances), np.array(mmd_mean), np.array(mmd_std))
            #distances_, mmd_ = np.array(distances), np.array(mmd_mean)
            df = pd.DataFrame()
            df['MMD'] = mmd_
            df['distance'] = distances_
            df['model'] = l
            df['loss'] = loss
            df['class'] = cl
            if data.empty:
                data = df
            else:
                data = data.append(df)

# pdb.set_trace()
g = sns.FacetGrid(data, row='loss', col='class', sharey=False, margin_titles=True)
g = (g.map(sns.lineplot, 'distance', 'MMD', 'model').add_legend())
# plt.subplots_adjust(right=0.8)
# g = sns.catplot(x='distance', y='MMD', hue='model', row='loss', col='class', kind='point', data=data, sharey=False)
# g.despine(left=True)
# plt.tight_layout()
# plt.show()
fig = plt.gcf()
fig.subplots_adjust(right=0.88)
fig.savefig(os.path.join(config.outpath, f'mog_mmd.pdf'))

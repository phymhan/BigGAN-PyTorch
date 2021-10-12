import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pdb

def replicate_std(idx, mean, std):
    idx_ = np.concatenate((idx, idx, idx), 0)
    mean_ = np.concatenate((mean-std, mean, mean+std), 0)
    return idx_, mean_


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='nec')
parser.add_argument('--config', type=str, default='av200a')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--outpath', type=str, default='../figs')
parser.add_argument('--no_fig', action='store_true')
parser.add_argument('--which_best', type=str, default='FID')
parser.add_argument('--global_best', action='store_true')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
elif config.os == 'nec':
    log_root = '/Users/lhan/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/mlfs/active/BigGAN/logs'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/logs'


if config.config == 'ac100a':
    models = [
        'aaai_C100_proj',
        'aaai_C100_tac',
        'aaai_C100_fc',
        'aaai_C100_HY',
        'aaai_C100_adaptive-HY',
        'aaai_C100_scalar-HY',
        'aaai_C100_scalar-HY-penalty'
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
        'scalar',
        'scalar-penalty',
    ]
    step_size = 1000
    exp = config.config
    trim = 0
elif config.config == 'ai64a':
    models = [
        'aaai_I64_proj',
        'aaai_I64_tac',
        'aaai_I64_fc',
        'aaai_I64_HY',
        'aaai_I64_adaptive-HY',
        'aaai_I64_adaptive-HY_smd',
        'aaai_I64_scalar-HY_smd',
        'aaai_I64_scalar-HY_smd-penalty'
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'sigma-hybrid',
        'adaptive-hybrid',
        'scalar',
        'scalar-penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'ai64':
    models = [
        'aaai_I64_proj',
        'aaai_I64_tac',
        'aaai_I64_fc',
        'aaai_I64_HY',
        'aaai_I64_adaptive-HY_smd',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'av200a':
    models = [
        'aaai_V200_proj',
        'aaai_V200_tac',
        'aaai_V200_fc',
        'aaai_V200_HY',
        'aaai_V200_adaptive-HY',
        'aaai_V200_scalar-HY',
        'aaai_V200_scalar-HY-penalty'
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
        'scalar',
        'scalar-penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 28
elif config.config == 'av200':
    models = [
        'aaai_V200_proj',
        'aaai_V200_tac',
        'aaai_V200_fc',
        'aaai_V200_HY',
        'aaai_V200_adaptive-HY',
        # 'aaai_V200_scalar-HY-penalty',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
    ]
    step_size = 2000
    exp = config.config
    trim = 28
elif config.config == 'av500a':
    models = [
        'aaai_V500_proj',
        'aaai_V500_tac',
        'aaai_V500_fc',
        'aaai_V500_HY',
        'aaai_V500_adaptive-HY',
        'aaai_V500_scalar-HY',
        'aaai_V500_scalar-HY-penalty'
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
        'scalar',
        'scalar-penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 50
elif config.config == 'av500':
    models = [
        'aaai_V500_proj',
        'aaai_V500_tac',
        'aaai_V500_fc',
        'aaai_V500_HY',
        # 'aaai_V500_scalar-HY-penalty',
        'aaai_V500_adaptive-HY',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
    ]
    step_size = 2000
    exp = config.config
    trim = 50
else:
    raise NotImplementedError

prefix = config.prefix
no_fig = config.no_fig or prefix
best_itr = {}
if not no_fig:
    data_is = pd.DataFrame()
    data_fid = pd.DataFrame()
    data_meanfid = pd.DataFrame()
    data_maxfid = pd.DataFrame()
else:
    trim = 0

print(config)
print(models)
print(legend)
best_is = {}
best_fid = {}
best_fid_mean = {}
best_fid_max = {}

for m, l in zip(models, legend):
    IS_mean = np.load(os.path.join(log_root, m, f'{prefix}IS_mean.npy'))
    IS_std = np.load(os.path.join(log_root, m, f'{prefix}IS_std.npy'))
    n = min(trim, len(IS_mean)) if trim > 0 else len(IS_mean)
    # print(m)
    # print(n)
    if config.which_best == 'IS':
        IS = np.load(os.path.join(log_root, m, f'{prefix}IS_mean.npy'))
        IS = IS[:n]
        j = np.argmax(IS)
    elif config.which_best == 'FID':
        FID = np.load(os.path.join(log_root, m, f'{prefix}FID.npy'))
        FID = FID[:n]
        j = np.argmin(FID)
    elif config.which_best == 'meanFID':
        IntraFID = np.load(os.path.join(log_root, m, f'{prefix}IntraFID.npy'))
        IntraFID = IntraFID[:n,...]
        FID_mean = np.mean(IntraFID, axis=1)
        j = np.argmin(FID_mean)
    elif config.which_best == 'maxFID':
        IntraFID = np.load(os.path.join(log_root, m, f'{prefix}IntraFID.npy'))
        IntraFID = IntraFID[:n,...]
        FID_max = np.max(IntraFID, axis=1)
        j = np.argmin(FID_max)
    else:
        raise NotImplementedError
    best_itr[l] = j * step_size
    FID = np.load(os.path.join(log_root, m, f'{prefix}FID.npy'))
    IntraFID = np.load(os.path.join(log_root, m, f'{prefix}IntraFID.npy'))
    FID_mean = np.mean(IntraFID, axis=1)
    FID_max = np.max(IntraFID, axis=1)
    # best_fid_mean[l] = np.min(FID_mean)
    # best_fid_max[l] = np.min(FID_max)
    best_fid_mean[l] = FID_mean[j]
    best_fid_max[l] = FID_max[j]
    if config.global_best:
        best_is[l] = (IS_mean[:n].max(), IS_std[IS_mean[:n].argsort()[-1]])
        best_fid[l] = FID.min()
    else:
        best_is[l] = (IS_mean[j], IS_std[j])
        best_fid[l] = FID[j]
    # print(f'=> {l} achieves best intra-FID {FID_mean.min()} and minimum max intra-FID {FID_max.min()}.')

    if not no_fig:
        iteration = np.arange(1, n+1) * step_size
        iteration_, IS_ = replicate_std(iteration, IS_mean[:n], IS_std[:n])

        df = pd.DataFrame()
        df['IS'] = IS_
        df['iteration'] = iteration_
        df['model'] = l
        if data_is.empty:
            data_is = df
        else:
            data_is = data_is.append(df)
        
        df = pd.DataFrame()
        df['FID'] = FID[:n]
        df['iteration'] = iteration
        df['model'] = l
        if data_fid.empty:
            data_fid = df
        else:
            data_fid = data_fid.append(df)
        
        df = pd.DataFrame()
        df['mean-FID'] = FID_mean[:n]
        df['iteration'] = iteration
        df['model'] = l
        if data_meanfid.empty:
            data_meanfid = df
        else:
            data_meanfid = data_meanfid.append(df)
        
        df = pd.DataFrame()
        df['max-FID'] = FID_max[:n]
        df['iteration'] = iteration
        df['model'] = l
        if data_maxfid.empty:
            data_maxfid = df
        else:
            data_maxfid = data_maxfid.append(df)

best_is_list = [best_is[l][0] for l in legend]
best_fid_list = [best_fid[l] for l in legend]
best_method_is = legend[best_is_list.index(max(best_is_list))]
best_method_fid = legend[best_fid_list.index(min(best_fid_list))]
best_mean_list = [best_fid_mean[l] for l in legend]
best_max_list = [best_fid_max[l] for l in legend]
best_method_mean = legend[best_mean_list.index(min(best_mean_list))]
best_method_max = legend[best_max_list.index(min(best_max_list))]
print(legend)
is_str = {l: f' {best_is[l][0]:.2f} $\pm$ {best_is[l][1]:.2f} ' for l in legend}
fid_str = {l: f' {best_fid[l]:.2f} ' for l in legend}
mean_str = {l: f' {best_fid_mean[l]:.2f} ' for l in legend}
max_str = {l: f' {best_fid_max[l]:.2f} ' for l in legend}
print(f'=> {best_method_is} achieves highest IS.')
print(f'=> {best_method_fid} achieves lowest FID.')
print(f'=> {best_method_mean} achieves minimum intra-FID.')
print(f'=> {best_method_max} achieves minimum max intra-FID.')
## reporting
# plain
print(f'IS:')
print(is_str)
print(f'FID:')
print(fid_str)
print(f'Mean FID:')
print(mean_str)
print(f'Max FID:')
print(max_str)
print(f'Iteration-{config.which_best}:')
print(best_itr)
is_str[best_method_is] = ' {\\bf' + is_str[best_method_is] + '} '
fid_str[best_method_fid] = ' {\\bf' + fid_str[best_method_fid] + '} '
mean_str[best_method_mean] = ' {\\bf' + mean_str[best_method_mean] + '} '
max_str[best_method_max] = ' {\\bf' + max_str[best_method_max] + '} '
print('=' * 100)
# print(' & '.join([is_str[l]+'&'+fid_str[l]+'&'+mean_str[l]+'&'+max_str[l] for l in legend]))
print(' & '.join([is_str[l]+'&'+fid_str[l]+'&'+max_str[l] for l in legend]))

if not no_fig:
    for size in [(4,3)]:
        sns.set()
        sns.set_context('paper')
        fig = plt.figure(figsize=size)
        sns.lineplot(x='iteration', y='IS', hue='model', data=data_is)
        plt.tight_layout()
        fig.savefig(os.path.join(config.outpath, f'{exp}_is_{size[0]}x{size[1]}.pdf'))
        sns.set_context('paper')
        fig = plt.figure(figsize=size)
        sns.lineplot(x='iteration', y='FID', hue='model', data=data_fid)
        plt.tight_layout()
        fig.savefig(os.path.join(config.outpath, f'{exp}_fid_{size[0]}x{size[1]}.pdf'))
        sns.set_context('paper')
        fig = plt.figure(figsize=size)
        sns.lineplot(x='iteration', y='mean-FID', hue='model', data=data_meanfid)
        plt.tight_layout()
        fig.savefig(os.path.join(config.outpath, f'{exp}_meanfid_{size[0]}x{size[1]}.pdf'))
        sns.set_context('paper')
        fig = plt.figure(figsize=size)
        sns.lineplot(x='iteration', y='max-FID', hue='model', data=data_maxfid)
        plt.tight_layout()
        fig.savefig(os.path.join(config.outpath, f'{exp}_maxfid_{size[0]}x{size[1]}.pdf'))
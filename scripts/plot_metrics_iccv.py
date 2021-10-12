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
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='c100')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--outpath', type=str, default='../figs_iccv')
parser.add_argument('--no_fig', action='store_true')
parser.add_argument('--which_best', type=str, default='FID')
parser.add_argument('--global_best', action='store_true')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
elif config.os == 'nec':
    # log_root = '/Users/lhan/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/mlfs/active/BigGAN/logs'
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/results_aaai/logs'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/logs'


if config.config == 'c100':
    models = [
        'iccv_C100_proj',
        'iccv_C100_tac',
        'iccv_C100_fc',
        'iccv_C100_p2',
        'iccv_C100_p2ap',
    ]
    legend = [
        'Proj-GAN',
        'TAC-GAN',
        'f-cGAN',
        'P2GAN',
        'P2GAN-w',
    ]
    step_size = 2000
    exp = config.config
    trim = 0
elif config.config == 'c100ab':
    models = [
        'iccv_C100_proj',
        'iccv_C100_p2',
        # 'iccv_C100_p2ap',
        'iccv_C100_over',
        'iccv_C100_psi0',
        'iccv_C100_na',
    ]
    legend = [
        'Proj-GAN',
        'P2GAN',
        # 'P2GAN-w',
        '$\lambda \equiv 0$',
        '$\psi \equiv 0$',
        'Naive',
    ]
    step_size = 2000
    exp = config.config
    trim = 30
elif config.config == 'c100w':
    models = [
        'iccv_C100_p2',
        'iccv_C100_p2_decay=0.995',
        'iccv_C100_p2_decay=0.9995',
        'iccv_C100_p2s',
        'iccv_C100_p2sp',
        'iccv_C100_p2a',
        'iccv_C100_p2ap',
    ]
    legend = [
        'P2GAN',
        'P2GAN-d (T=200)',
        'P2GAN-d (T=2000)',
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-a',
        'P2GAN-ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 30
elif config.config == 'i64':
    # aaai
    models = [
        'aaai_I64_proj',
        'aaai_I64_tac',
        'aaai_I64_fc',
        'aaai_I64_HY',
        'aaai_I64_adaptive-HY_smd',
    ]
    legend = [
        'Proj-GAN',
        'TAC-GAN',
        'f-cGAN',
        'P2GAN',
        'P2GAN-w',
    ]
    step_size = 2000
    exp = config.config
    trim = 40 #50
elif config.config == 'i64ab':
    # aaai and nips
    # models = [
    #     '../results_aaai/logs/aaai_I64_proj',
    #     '../results_aaai/logs/aaai_I64_HY',
    #     '../results_aaai/logs/aaai_I64_adaptive-HY_smd',
    #     '../results_nips/logs_old/final_I64_over_2gpu_0',
    #     '../results_nips/logs_old/final_I64_proj+nofc_2gpu_0',
    # ]
    # nips
    models = [
        '../results_nips/logs_old/final_I64_proj_2gpu_0',
        '../results_nips/logs_old/final_I64_HY_2gpu_0',
        '../results_nips/logs_old/final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
        '../results_nips/logs_old/final_I64_over_2gpu_0',
        '../results_nips/logs_old/final_I64_proj+nofc_2gpu_0',
    ]
    legend = [
        'Proj-GAN',
        'P2GAN',
        'P2GAN-w',
        '$\lambda \equiv 0$',
        '$\psi \equiv 0$',
    ]
    step_size = 2000
    exp = config.config
    trim = 32
elif config.config == 'i64w':
    # aaai
    models = [
        '../results_aaai/logs/aaai_I64_HY',
        '../results_aaai/logs/aaai_I64_scalar-HY_smd',
        '../results_aaai/logs/aaai_I64_scalar-HY_smd-penalty',
        '../results_aaai/logs/aaai_I64_adaptive-HY_smd',
    ]
    legend = [
        'P2GAN',
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'i64wo':
    # nips
    models = [
        '../results_nips/logs_old/final_I64_HY_2gpu_0',
        '../results_aaai/logs/aaai_I64_scalar-HY_smd',
        '../results_aaai/logs/aaai_I64_scalar-HY_smd-penalty',
        '../results_nips/logs_old/final_I64_HY+sigmoid_detach_dwl_2gpu_1',
        '../results_nips/logs_old/final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    ]
    legend = [
        'P2GAN',
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-a',
        'P2GAN-ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'i64db':
    # aaai and nips
    # models = [
    #     '../results_aaai/logs/aaai_I64_HY',
    #     '../results_nips/logs_old/final_I64_HY_2gpu_0',
    #     '../results_aaai/logs/aaai_I64_adaptive-HY_smd',
    #     '../results_nips/logs_old/final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    # ]
    # legend = [
    #     'aaai_p2',
    #     'nips_p2',
    #     'aaai_p2ap',
    #     'nips_p2ap',
    # ]
    models = [
        '../results_aaai/logs/aaai_I64_proj',
        '../results_nips/logs_old/final_I64_proj_2gpu_0',
        '../results_aaai/logs/aaai_I64_adaptive-HY_smd',
        '../results_nips/logs_old/final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    ]
    legend = [
        'aaai_proj',
        'nips_proj',
        'aaai_p2ap',
        'nips_p2ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 0
elif config.config == 'i64o':
    # old
    models = [
        '../results_nips/logs_old/final_I64_proj_2gpu_0',
        '../results_nips/logs_old/final_I64_tac_2gpu_1',
        '../results_nips/logs_old/final_I64_fc+revkl_2gpu_0',
        '../results_nips/logs_old/final_I64_HY_2gpu_0',
        '../results_nips/logs_old/final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    ]
    legend = [
        'proj',
        'tac',
        'fc',
        'p2',
        'p2ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 60
elif config.config == 'v200':
    models = [
        'final_V200_proj',
        'final_V200_tac',
        'final_V200_fc',
        'final_V200_p2',
        'final_V200_p2ap',
    ]
    legend = [
        'Proj-GAN',
        'TAC-GAN',
        'f-cGAN',
        'P2GAN',
        'P2GAN-w',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'v200ab':
    models = [
        'final_V200_proj',
        'final_V200_p2',
        'final_V200_over',
        'final_V200_psi0',
        'final_V200_na',
    ]
    legend = [
        'Proj-GAN',
        'P2GAN',
        # 'P2GAN-w',
        '$\lambda \equiv 0$',
        '$\psi \equiv 0$',
        'Naive',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'v200w':
    models = [
        'final_V200_p2',
        'final_V200_p2_decay=0.995',
        'final_V200_p2_decay=0.9995',
        'final_V200_p2s_1',
        'final_V200_p2sp_1',
        'final_V200_p2a',
        'final_V200_p2ap',
    ]
    legend = [
        'P2GAN',
        'P2GAN-d (T=200)',
        'P2GAN-d (T=2000)',
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-a',
        'P2GAN-ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 25
elif config.config == 'v500':
    models = [
        'final_V500_proj',
        'final_V500_tac',
        'final_V500_fc',
        'final_V500_p2',
        'final_V500_p2ap',
    ]
    legend = [
        'Proj-GAN',
        'TAC-GAN',
        'f-cGAN',
        'P2GAN',
        'P2GAN-w',
    ]
    step_size = 2000
    exp = config.config
    trim = 50
else:
    raise NotImplementedError

prefix = ''
no_fig = False
no_intra_fid = True
# no_fig = config.no_fig or prefix
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
if not no_intra_fid:
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
    if not no_intra_fid:
        IntraFID = np.load(os.path.join(log_root, m, f'{prefix}IntraFID.npy'))
        FID_mean = np.mean(IntraFID, axis=1)
        FID_max = np.max(IntraFID, axis=1)
        # best_fid_mean[l] = np.min(FID_mean)
        # best_fid_max[l] = np.min(FID_max)
        best_fid_mean[l] = FID_mean[j]
        best_fid_max[l] = FID_max[j]
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
        
        if not no_intra_fid:
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
if not no_intra_fid:
    best_mean_list = [best_fid_mean[l] for l in legend]
    best_max_list = [best_fid_max[l] for l in legend]
    best_method_mean = legend[best_mean_list.index(min(best_mean_list))]
    best_method_max = legend[best_max_list.index(min(best_max_list))]
print(legend)
is_str = {l: f' {best_is[l][0]:.2f} $\pm$ {best_is[l][1]:.2f} ' for l in legend}
fid_str = {l: f' {best_fid[l]:.2f} ' for l in legend}
if not no_intra_fid:
    mean_str = {l: f' {best_fid_mean[l]:.2f} ' for l in legend}
    max_str = {l: f' {best_fid_max[l]:.2f} ' for l in legend}
print(f'=> {best_method_is} achieves highest IS.')
print(f'=> {best_method_fid} achieves lowest FID.')
if not no_intra_fid:
    print(f'=> {best_method_mean} achieves minimum intra-FID.')
    print(f'=> {best_method_max} achieves minimum max intra-FID.')
## reporting
# plain
print(f'IS:')
print(is_str)
print(f'FID:')
print(fid_str)
if not no_intra_fid:
    print(f'Mean FID:')
    print(mean_str)
    print(f'Max FID:')
    print(max_str)
print(f'Iteration-{config.which_best}:')
print(best_itr)
is_str[best_method_is] = ' {\\bf' + is_str[best_method_is] + '} '
fid_str[best_method_fid] = ' {\\bf' + fid_str[best_method_fid] + '} '
if not no_intra_fid:
    mean_str[best_method_mean] = ' {\\bf' + mean_str[best_method_mean] + '} '
    max_str[best_method_max] = ' {\\bf' + max_str[best_method_max] + '} '
print('=' * 100)
# print(' & '.join([is_str[l]+'&'+fid_str[l]+'&'+mean_str[l]+'&'+max_str[l] for l in legend]))
if not no_intra_fid:
    print(' & '.join([is_str[l]+'&'+fid_str[l]+'&'+max_str[l] for l in legend]))
size = (4,3)
if not no_fig:
    sns.set()
    sns.set_context('paper')
    fig = plt.figure(figsize=size)
    sns.lineplot(x='iteration', y='IS', hue='model', data=data_is)
    plt.tight_layout()
    fig.savefig(os.path.join(config.outpath, f'curve_{exp}_is_iccv.pdf'))
    sns.set_context('paper')
    fig = plt.figure(figsize=size)
    sns.lineplot(x='iteration', y='FID', hue='model', data=data_fid)
    plt.tight_layout()
    fig.savefig(os.path.join(config.outpath, f'curve_{exp}_fid_iccv.pdf'))
    # sns.set_context('paper')
    # fig = plt.figure(figsize=size)
    # sns.lineplot(x='iteration', y='mean-FID', hue='model', data=data_meanfid)
    # plt.tight_layout()
    # fig.savefig(os.path.join(config.outpath, f'curve_{exp}_meanfid_iccv.pdf'))
    # sns.set_context('paper')
    # fig = plt.figure(figsize=size)
    # sns.lineplot(x='iteration', y='max-FID', hue='model', data=data_maxfid)
    # plt.tight_layout()
    # fig.savefig(os.path.join(config.outpath, f'curve_{exp}_maxfid_iccv.pdf'))
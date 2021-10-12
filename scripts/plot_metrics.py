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
parser.add_argument('--config', type=str, default='mine')
parser.add_argument('--outpath', type=str, default='../figs')
parser.add_argument('--no_fig', action='store_true')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/logs'

if config.config == 'mine':
    models = [
        'which_mi_C100IB_ce+ce',
        'which_mi_C100IB_ce+mi',
        'which_mi_C100IB_ce+sm',
        'which_mi_C100IB_mi+mi',
        'which_mi_C100IB_sm+sm',
        'which_mi_C100IB_mi+ce',
        'which_mi_C100IB_sm+ce']
    legend = [
        'CE + CE',
        'CE + MINE',
        'CE + MINE-LSE',
        'MINE + MINE',
        'MINE-LSE + MINE-LSE',
        'MINE + CE',
        'MINE-LSE + CE'
    ]
    step_size = 2000
    exp = config.config
    trim = 0
elif config.config == 'f-div_v200':
    models = [
        'zzzffftest_v200_simple+revkl_2gpu_0',
        'zzzffftest_v200_simple+kl_2gpu_0',
        'zzzffftest_v200_simple+pearson_2gpu_0',
        'zzzffftest_v200_simple+squared_2gpu_0',
        'zzzffftest_v200_simple+jsd_2gpu_0',
        'zzzffftest_v200_simple+gan_2gpu_0',
    ]
    legend = [
        'Reverse KL',
        'KL',
        'Pearson',
        'Squared',
        'JSD',
        'GAN',
    ]
    step_size = 1000
    exp = config.config
    trim = 80
elif config.config == 'f-div_c100ib':
    models = [
        'final_C100IB_fc+revkl_2gpu_0',
        'final_C100IB_fc+kl_2gpu_0',
        'final_C100IB_fc+pearson_2gpu_0',
        'final_C100IB_fc+squared_2gpu_0',
        'final_C100IB_fc+jsd_2gpu_0',
        'final_C100IB_fc+gan_2gpu_0'
    ]
    legend = [
        'Reverse KL',
        'KL',
        'Pearson',
        'Squared',
        'JSD',
        'GAN'
    ]
    step_size = 500
    exp = config.config
    trim = 0
elif config.config == 'nofc':
    models = [
        'c100ib_proj_2',
        'c100ib_proj_nofc',
        'c100ib_ac',
    ]
    legend = [
        'projection',
        'projection w/o $\psi$',
        'AC-GAN',
    ]
    step_size = 250
    exp = config.config
    trim = 0
elif config.config == 'c100ib':
    models = [
        'final_C100IB_proj_2gpu_0',
        'final_C100IB_tac_2gpu_0',
        'final_C100IB_fc+revkl_2gpu_0',
        'final_C100IB_HY_2gpu_0',
        'final_C100IB_HY+penalty_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
    ]
    step_size = 500
    exp = config.config
    trim = 0
elif config.config == 'c100':
    models = [
        'final_C100_proj_2gpu_0',
        'final_C100_tac_2gpu_0',
        'final_C100_fc+revkl_2gpu_0',
        'final_C100_HY_2gpu_0',
        'final_C100_HY+penalty_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
    ]
    step_size = 500
    exp = config.config
    trim = 0
elif config.config == 'c100n':
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
elif config.config == 'i64':
    models = [
        'final_I64_proj_2gpu_0',
        'final_I64_tac_2gpu_1',
        'final_I64_fc+revkl_2gpu_0',
        'final_I64_HY_2gpu_0',
        'final_I64_HY+sigmoid_detach_dwl_2gpu_1',
        'final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive',
        'adaptive-penalty'
    ]
    step_size = 2000
    exp = config.config
    trim = 60
elif config.config == 'v200a':
    models = [
        'final_V200_proj_2gpu_0',
        'final_V200_tac_2gpu_0',
        'final_V200_fc+revkl_2gpu_0',
        'final_V200_over_2gpu_0',
        'final_V200_HY_2gpu_0',
        'final_V200_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V200_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        '$\lambda \equiv 1$',
        'hybrid',
        'adaptive w/o penalty',
        'adaptive w/ penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 40
elif config.config == 'v500a':
    models = [
        'final_V500_proj_2gpu_0',
        'final_V500_tac_2gpu_1',
        'final_V500_fc+revkl_2gpu_0',
        'final_V500_over_2gpu_0',
        'final_V500_HY_2gpu_0',
        'final_V500_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V500_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        '$\lambda \equiv 1$',
        'hybrid',
        'adaptive w/o penalty',
        'adaptive w/ penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 60
elif config.config == 'v1000a':
    models = [
        'final_V1000_proj_2gpu_0',
        'final_V1000_tac_2gpu_0',
        'final_V1000_fc+revkl_2gpu_0',
        'final_V1000_over_2gpu_0',
        'final_V1000_HY_2gpu_0',
        'final_V1000_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V1000_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        '$\lambda \equiv 1$',
        'hybrid',
        'adaptive w/o penalty',
        'adaptive w/ penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 60
elif config.config == 'v200':
    models = [
        'final_V200_proj_2gpu_0',
        'final_V200_tac_2gpu_0',
        'final_V200_fc+revkl_2gpu_0',
        'final_V200_HY_2gpu_0',
        'final_V200_HY+sigmoid_penalty_detach_dwl_2gpu_0',
        'final_V200_over_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive-hybrid',
        '$\lambda \equiv 1$',
    ]
    step_size = 2000
    exp = config.config
    trim = 30
elif config.config == 'v500':
    models = [
        'final_V500_proj_2gpu_0',
        'final_V500_tac_2gpu_1',
        'final_V500_fc+revkl_2gpu_0',
        'final_V500_HY_2gpu_0',
        'final_V500_HY+sigmoid_penalty_detach_dwl_2gpu_0',
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
    trim = 35
elif config.config == 'v1000':
    models = [
        'final_V1000_proj_2gpu_0',
        'final_V1000_tac_2gpu_0',
        'final_V1000_fc+revkl_2gpu_0',
        'final_V1000_HY_2gpu_0',
        'final_V1000_HY+sigmoid_penalty_detach_dwl_2gpu_0',
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

data_is = pd.DataFrame()
data_fid = pd.DataFrame()
print(config)
print(models)
print(legend)
best_is = {}
best_fid = {}
for m, l in zip(models, legend):
    IS_mean = np.load(os.path.join(log_root, m, 'IS_mean.npy'))
    IS_std = np.load(os.path.join(log_root, m, 'IS_std.npy'))
    # print(m, IS_mean)
    n = min(trim, len(IS_mean)) if trim > 0 else len(IS_mean)
    best_is[l] = (IS_mean[:n].max(), IS_std[IS_mean.argsort()[-1]])
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

    FID = np.load(os.path.join(log_root, m, 'FID.npy'))
    n = min(trim, len(FID)) if trim > 0 else len(FID)
    best_fid[l] = FID[:n].min()
    iteration = np.arange(1, n+1) * step_size
    df = pd.DataFrame()
    df['FID'] = FID[:n]
    df['iteration'] = iteration
    df['model'] = l
    if data_fid.empty:
        data_fid = df
    else:
        data_fid = data_fid.append(df)
    print(f'=> {l} achieves best IS {IS_mean[:n].max()} and best FID {FID[:n].min()}.')


best_is_list = [best_is[l][0] for l in legend]
best_fid_list = [best_fid[l] for l in legend]
best_method_is = legend[best_is_list.index(max(best_is_list))]
best_method_fid = legend[best_fid_list.index(min(best_fid_list))]
print(legend)
is_str = {l: f' {best_is[l][0]:.2f} $\pm$ {best_is[l][1]:.2f} ' for l in legend}
fid_str = {l: f' {best_fid[l]:.2f} ' for l in legend}
is_str[best_method_is] = ' {\\bf' + is_str[best_method_is] + '} '
fid_str[best_method_fid] = ' {\\bf' + fid_str[best_method_fid] + '} '
print(f'=> {best_method_is} achieves highest IS.')
print(f'=> {best_method_fid} achieves lowest FID.')
## reporting
# IS and FID in one table:
# plain
print('=' * 100)
print(' & '.join([f' {best_is[l][0]:.2f} $\\pm$ {best_is[l][1]:.2f} & {best_fid[l]:.2f} ' for l in legend]))
# with bf
print('-' * 100)
print(' & '.join([is_str[l]+'&'+fid_str[l] for l in legend]))

# IS and FID in two tables:
# plain
print('=' * 100)
print(' & '.join([f' {best_is[l][0]:.2f} $\\pm$ {best_is[l][1]:.2f} ' for l in legend]))
print('-' * 100)
print(' & '.join([f' {best_fid[l]:.2f} ' for l in legend]))
print('-' * 100)
print(' & '.join([is_str[l] for l in legend]))
print(' & '.join([fid_str[l] for l in legend]))

if not config.no_fig:
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

# sns.set()
# sns.set_context('paper')
# fig, ax = plt.subplots(1, 2)
# plt.subplot(1, 2, 1)
# sns.lineplot(x='iteration', y='IS', hue='model', data=data_is)
# # plt.show()
# # fig.savefig('IS_plot.pdf')
# plt.subplot(1, 2, 2)
# sns_plot = sns.lineplot(x='iteration', y='FID', hue='model', data=data_fid)
# # plt.show()
# # sns_plot.savefig('FID_plot.pdf')
# plt.show()

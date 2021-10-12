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
parser.add_argument('--which_best', type=str, default='FID')
parser.add_argument('--global_best', action='store_true')
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
elif config.os == 'nec':
    log_root = '/Users/lhan/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/mlfs/active/BigGAN/logs'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/logs'

if config.config == 'new_c100ib':
    models = [
        'new2_final_C100IB_proj_2gpu_0',
        'new2_final_C100IB_tac_2gpu_0',
        'new2_final_C100IB_fc+revkl_2gpu_0',
        'new2_final_C100IB_HY_2gpu_0',
        'new_final_C100IB_HY+penalty_2gpu_0',
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
elif config.config == 'new_c100':
    models = [
        'new2_final_C100_proj_2gpu_0',
        'new2_final_C100_tac_2gpu_0',
        'new2_final_C100_fc+revkl_2gpu_0',
        'new2_final_C100_HY_2gpu_0',
        'new2_final_C100_HY+penalty_2gpu_0',
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
elif config.config == 'i64':
    models = [
        'final_I64_proj_2gpu_0',
        'final_I64_tac_2gpu_1',
        'final_I64_fc+revkl_2gpu_0',
        'final_I64_HY_2gpu_0',
        # 'final_I64_HY+sigmoid_detach_dwl_2gpu_1',
        'final_I64_HY+sigmoid_penalty_detach_dwl_2gpu_1',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        # 'adaptive',
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
        'final_V200_HY_2gpu_0',
        # 'final_V200_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V200_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        # 'adaptive w/o penalty',
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
        'final_V500_HY_2gpu_0',
        # 'final_V500_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V500_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        # 'adaptive w/o penalty',
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
        'final_V1000_HY_2gpu_0',
        'final_V1000_HY+sigmoid_detach_dwl_2gpu_0',
        # 'final_V1000_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'Projection',
        'TAC-GAN',
        'f-cGAN',
        'hybrid',
        'adaptive w/o penalty',
        # 'adaptive w/ penalty',
    ]
    step_size = 2000
    exp = config.config
    trim = 60
else:
    raise NotImplementedError

print(config)
print(models)
print(legend)
best_is = {}
best_fid = {}
best_fid_mean = {}
best_fid_max = {}
for m, l in zip(models, legend):
    if config.which_best == 'IS':
        IS = np.load(os.path.join(log_root, m, 'test_IS_mean.npy'))
        j = np.argmax(IS)
    elif config.which_best == 'FID':
        FID = np.load(os.path.join(log_root, m, 'test_FID.npy'))
        j = np.argmin(FID)
    elif config.which_best == 'meanFID':
        IntraFID = np.load(os.path.join(log_root, m, 'test_IntraFID.npy'))
        FID_mean = np.mean(IntraFID, axis=1)
        j = np.argmin(FID_mean)
    elif config.which_best == 'maxFID':
        IntraFID = np.load(os.path.join(log_root, m, 'test_IntraFID.npy'))
        FID_max = np.max(IntraFID, axis=1)
        j = np.argmin(FID_max)
    else:
        raise NotImplementedError
    IS_mean = np.load(os.path.join(log_root, m, 'test_IS_mean.npy'))
    IS_std = np.load(os.path.join(log_root, m, 'test_IS_std.npy'))
    FID = np.load(os.path.join(log_root, m, 'test_FID.npy'))
    IntraFID = np.load(os.path.join(log_root, m, 'test_IntraFID.npy'))
    FID_mean = np.mean(IntraFID, axis=1)
    FID_max = np.max(IntraFID, axis=1)
    # best_fid_mean[l] = np.min(FID_mean)
    # best_fid_max[l] = np.min(FID_max)
    best_fid_mean[l] = FID_mean[j]
    best_fid_max[l] = FID_max[j]
    if config.global_best:
        best_is[l] = (IS_mean.max(), IS_std[IS_mean.argsort()[-1]])
        best_fid[l] = FID.min()
    else:
        best_is[l] = (IS_mean[j], IS_std[j])
        best_fid[l] = FID[j]
    # print(f'=> {l} achieves best intra-FID {FID_mean.min()} and minimum max intra-FID {FID_max.min()}.')

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
is_str[best_method_is] = ' {\\bf' + is_str[best_method_is] + '} '
fid_str[best_method_fid] = ' {\\bf' + fid_str[best_method_fid] + '} '
mean_str[best_method_mean] = ' {\\bf' + mean_str[best_method_mean] + '} '
max_str[best_method_max] = ' {\\bf' + max_str[best_method_max] + '} '
print('=' * 100)
print(' & '.join([is_str[l]+'&'+fid_str[l]+'&'+mean_str[l]+'&'+max_str[l] for l in legend]))

import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pdb
st = pdb.set_trace

def replicate_std(idx, mean, std):
    idx_ = np.concatenate((idx, idx, idx), 0)
    mean_ = np.concatenate((mean-std, mean, mean+std), 0)
    return idx_, mean_

def read_array_from_log(fp):
    with open(fp, 'r') as f:
        lines = f.readlines()
    is_score = []
    fid_score = []
    has_new_entry = False
    step_count = -1
    for line in lines:
        if 'Start Evaluation (' in line:
            curr_step = int(line[line.find('Start Evaluation (')+18:line.find(' Step)')])
            has_new_entry = curr_step > step_count
            step_count = curr_step
        if has_new_entry and '> FID score (Step:' in line:
            value = float(line.split()[-1])
            fid_score.append(value)
        if has_new_entry and '> Inception score (Step:' in line:
            value = float(line.split()[-1])
            is_score.append(value)
            has_new_entry = False
    return np.array(is_score), np.array(fid_score)


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='p2ap')
parser.add_argument('--outpath', type=str, default='../figs_iccv')
parser.add_argument('--trim', type=int, default=10000)
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/PyTorch-StudioGAN/logs'

if config.config == 'p2ap':
    legend = [
        'Proj-GAN',
        'ContraGAN',
        'P2GAN-ap',
    ]
    step_size = 2000
    exp = config.config
    trim = 0
    train_log = [
        'IMAGENET/BigGAN256-train-2021_01_24_03_52_15.log',
        'IMAGENET/ContraGAN256-train-2021_01_25_13_55_18.log',
        'I128_p2ap-train-2021_03_22_04_23_46.log',
    ]
elif config.config == 'p2pd':
    legend = [
        'Proj-GAN',
        'Proj-Random',
        'Proj-Center',
    ]
    step_size = 2000
    exp = config.config
    trim = 0
    train_log = [
        'IMAGENET/BigGAN256-train-2021_01_24_03_52_15.log',
        'I128_proj-train-2021_03_27_17_54_14.log',
        'I128_proj_test-train-2021_03_27_00_35_12.log',
    ]
elif config.config == 'p2db':
    legend = [
        'Proj-GAN',
        'P2GAN-ap-exp',
        'P2GAN-ap-sgm',
        'P2GAN-s-exp',
    ]
    step_size = 2000
    exp = config.config
    trim = 0
    train_log = [
        'I128_proj_test-train-2021_03_27_00_35_12.log',
        'I128_p2ap-train-2021_03_22_04_23_46.log',
        'I128_p2ap-train-2021_03_28_19_42_36.log',
        'I128_p2s-train-2021_03_22_18_40_21.log',
    ]
else:
    raise NotImplementedError

data1 = pd.DataFrame()
data2 = pd.DataFrame()
print(config)
print(legend)
# from scipy import signal
# trim = config.trim
data_is = pd.DataFrame()
data_fid = pd.DataFrame()
for i, l in enumerate(legend):
    is_score, fid_score = read_array_from_log(os.path.join(log_root, train_log[i]))
    n = min(trim, len(is_score)) if trim > 0 else len(is_score)
    iteration = np.arange(1, n+1) * step_size

    df = pd.DataFrame()
    df['IS'] = is_score[:n]
    df['iteration'] = iteration
    df['model'] = l
    if data_is.empty:
        data_is = df
    else:
        data_is = data_is.append(df)
    
    df = pd.DataFrame()
    df['FID'] = fid_score[:n]
    df['iteration'] = iteration
    df['model'] = l
    if data_fid.empty:
        data_fid = df
    else:
        data_fid = data_fid.append(df)

if True:
    size = (4, 3)
    sns.set()
    sns.set_context('paper')
    fig = plt.figure(figsize=size)
    sns.lineplot(x='iteration', y='IS', hue='model', data=data_is)
    plt.tight_layout()
    fig.savefig(os.path.join(config.outpath, f'i128_{exp}_is_iccv.pdf'))
    sns.set_context('paper')
    fig = plt.figure(figsize=size)
    sns.lineplot(x='iteration', y='FID', hue='model', data=data_fid)
    plt.tight_layout()
    fig.savefig(os.path.join(config.outpath, f'i128_{exp}_fid_iccv.pdf'))

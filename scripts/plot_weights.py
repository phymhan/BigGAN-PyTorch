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

def read_array_from_log(fp):
    with open(fp, 'r') as f:
        return np.array([float(s.strip().split(':')[1]) for s in f.readlines()])


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='v200')
parser.add_argument('--outpath', type=str, default='../figs')
parser.add_argument('--trim', type=int, default=10000)
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/BigGAN-PyTorch/logs'
elif config.os == 'nec':
    log_root = '/Users/lhan/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/mlfs/active/BigGAN/logs'
else:
    # log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/results_nips/logs_old'
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/logs'

if config.config == 'wC100a':
    models = [
        'iccv_C100_p2a',
        'iccv_C100_p2ap',
        'iccv_C100_p2a_new',
        'iccv_C100_p2ap_new',
        # 'iccv_C100_p2s_1',
        # 'iccv_C100_p2sp_1',
    ]
    legend = [
        'P2GAN-a',
        'P2GAN-ap',
        'P2GAN-a_new',
        'P2GAN-ap_new',
        # 'P2GAN-s',
        # 'P2GAN-sp',
    ]
    transform = [
        lambda x: 1-x,
        lambda x: 1-x,
        lambda x: x,
        lambda x: x,
    ]
    step_size = 1
    exp = config.config
    trim = 200
    weight_log = [
        'weight_real.log',
        'weight_real.log',
        'weight_real.log',
        'weight_real.log',
        'weight_x.log',
        'weight_x.log',
    ]
elif config.config == 'wC100s':
    models = [
        'iccv_C100_p2s',
        'iccv_C100_p2sp',
        'iccv_C100_p2s_new',
        'iccv_C100_p2sp_new',
    ]
    legend = [
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-s_new',
        'P2GAN-sp_new',
    ]
    transform = [
        lambda x: 1-x,
        lambda x: 1-x,
        lambda x: 1/x-1,
        lambda x: 1/x-1,
    ]
    step_size = 1
    exp = config.config
    trim = 10000
    weight_log = [
        'weight_x.log',
        'weight_x.log',
        'weight_x.log',
        'weight_x.log',
    ]
elif config.config == 'wC100r':
    models = [
        'iccv_C100_p2ap',
        'iccv_C100_p2ap',
        'iccv_C100_p2ap_new',
        'iccv_C100_p2ap_new',
    ]
    legend = [
        'P2GAN-ap_real',
        'P2GAN-ap_fake',
        'P2GAN-ap_new_real',
        'P2GAN-ap_new_fake',
    ]
    transform = [
        lambda x: 1-x,
        lambda x: 1-x,
        lambda x: x,
        lambda x: x,
    ]
    step_size = 1
    exp = config.config
    trim = 200
    weight_log = [
        'weight_real.log',
        'weight_fake.log',
        'weight_real.log',
        'weight_fake.log',
    ]
elif config.config == 'wC100i':
    models = [
        'iccv_C100_p2ia',
        'iccv_C100_p2iap',
        'iccv_C100_p2is',
        'iccv_C100_p2isp',
    ]
    legend = [
        'P2GAN-ia',
        'P2GAN-iap',
        'P2GAN-is',
        'P2GAN-isp',
    ]
    transform = [
        lambda x: x,
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    step_size = 1
    exp = config.config
    trim = 10000
    weight_log = [
        'weight_real.log',
        'weight_real.log',
        'lambda_mi.log',
        'lambda_mi.log',
    ]
elif config.config == 'wV200a':
    models = [
        'final_V200_p2a',
        'final_V200_p2ap',
        'final_V200_p2ap_new',
    ]
    legend = [
        'P2GAN-a',
        'P2GAN-ap',
        'P2GAN-ap_new',
    ]
    transform = [
        lambda x: 1-x,
        lambda x: 1-x,
        lambda x: x,
    ]
    step_size = 1
    exp = config.config
    trim = 2000
    weight_log = [
        'weight_real.log',
        'weight_real.log',
        'weight_real.log',
    ]
elif config.config == 'wV200s':
    models = [
        'final_V200_p2s',
        'final_V200_p2sp',
        'final_V200_p2sp_new',
    ]
    legend = [
        'P2GAN-s',
        'P2GAN-sp',
        'P2GAN-sp_new',
    ]
    transform = [
        lambda x: 1-x,
        lambda x: 1-x,
        lambda x: 1/x-1,
    ]
    step_size = 1
    exp = config.config
    trim = 50000
    weight_log = [
        'weight_x.log',
        'weight_x.log',
        'weight_x.log',
    ]
elif config.config == 'wV200i':
    models = [
        'final_V200_p2iap',
        'final_V200_p2is',
        'final_V200_p2isp',
    ]
    legend = [
        'P2GAN-iap',
        'P2GAN-is',
        'P2GAN-isp',
    ]
    transform = [
        lambda x: x,
        lambda x: x,
        lambda x: x,
    ]
    step_size = 1
    exp = config.config
    trim = 50000
    weight_log = [
        'weight_real.log',
        'lambda_mi.log',
        'lambda_mi.log',
    ]
elif config.config == 'v200':
    models = [
        'final_V200_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V200_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'P2GAN-a',
        'P2GAN-ap',
    ]
    step_size = 1
    exp = config.config
    trim = 10000
    weight_log = 'weight_real.log'
elif config.config == 'v200s':
    models = [
        'aaai_V200_scalar-HY',
        'aaai_V200_scalar-HY-penalty',
    ]
    legend = [
        'P2GAN-s',
        'P2GAN-sp',
    ]
    step_size = 1
    exp = config.config
    trim = 20000
    weight_log = 'weight_x.log'
elif config.config == 'v500':
    models = [
        'final_V500_HY+sigmoid_detach_dwl_2gpu_0',
        'final_V500_HY+sigmoid_penalty_detach_dwl_2gpu_0',
    ]
    legend = [
        'adaptive w/o penalty',
        'adaptive w/ penalty',
    ]
    step_size = 1
    exp = config.config
    trim = 10000
    weight_log = 'weight_real.log'
elif config.config == 'v500n':
    models = [
        'aaai_V500_scalar-HY',
        'aaai_V500_scalar-HY-penalty',
    ]
    legend = [
        'adaptive w/o penalty',
        'adaptive w/ penalty',
    ]
    step_size = 1
    exp = config.config
    trim = 20000
    weight_log = 'weight_x.log'
elif config.config == 'c100ib':
    models = [
        'final_C100IB_HY+sigmoid_dwl_2gpu_0',
        'final_C100IB_HY+penalty_dwl_2gpu_0',
    ]
    legend = [
        'P2GAN-a',
        'P2GAN-ap',
    ]
    step_size = 1
    exp = config.config
    trim = 10000
    weight_log = 'weight_real.log'
elif config.config == 'c100ibs':
    models = [
        'aaai_C100IB_scalar-HY',
        'aaai_C100IB_scalar-HY-penalty',
    ]
    legend = [
        'P2GAN-s',
        'P2GAN-sp',
    ]
    step_size = 1
    exp = config.config
    trim = 20000
    weight_log = 'weight_x.log'
else:
    raise NotImplementedError

data1 = pd.DataFrame()
data2 = pd.DataFrame()
print(config)
print(models)
print(legend)
from scipy import signal
# trim = config.trim
for i, (m, l) in enumerate(zip(models, legend)):
    wx = read_array_from_log(os.path.join(log_root, m, weight_log[i]))
    n = min(trim, len(wx)) if trim > 0 else len(wx)
    df = pd.DataFrame()
    wt = wx[:n]
    wt_sm = signal.savgol_filter(wt, 53, 3)
    df['$\lambda$'] = transform[i](wt_sm)
    df['iteration'] = np.arange(1, n+1) * step_size
    df['model'] = l

    if data1.empty:
        data1 = df
    else:
        data1 = data1.append(df)

for size in [(4,3)]:
    sns.set()
    sns.set_context('paper')
    fig = plt.figure(figsize=size)
    sns.lineplot(x='iteration', y='$\lambda$', hue='model', data=data1)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(config.outpath, f'weight_{exp}_new.pdf'))

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

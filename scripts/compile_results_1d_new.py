import os
import numpy as np
import copy
import argparse
from mmd import mix_rbf_mmd2
import torch
import pdb
st = pdb.set_trace

# sigma_list = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
sigma_list = [1.0, 2.0, 5.0, 10.0]

def parse_(s):
    mean, std = s.strip().split(',')
    return float(mean), float(std)

def fix_file(lines):
    if len(lines) == 25:
        return lines
    if lines[len(lines)-25].startswith('proj:'):
        # pdb.set_trace()
        return lines[len(lines)-25:]
    else:
        pdb.set_trace()
        raise RuntimeError

parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='mog')
parser.add_argument('--outpath', type=str, default='../figs')
parser.add_argument('--loss', type=str, nargs='+', default=['bce'])
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/BigGAN/MOG/1D/'

methods = ['proj', 'tac', 'fc', 'p2', 'p2w']
methods_valid = ['proj', 'tac', 'fc', 'p2', 'p2w']
classes = ['0', '1', '2', 'm']
distances = [d/2.+1 for d in [0, 2, 4, 6, 8]]
n_runs = 10
res_full = ''
avg_rank = {m: [] for m in methods}
res_mmd = {m: {d: {c: (999, 999) for c in classes} for d in distances} for m in methods_valid}
bf_mmd = {m: {d: {c: False for c in classes} for d in distances} for m in methods_valid}
for loss in config.loss:
    for dist in distances: #[0, 2, 4, 6, 8]:
        # dist = dist/2.+1
        res = {m: {'0': [], '1': [], '2': [], 'm': []} for m in methods}
        res_full += f'experiment: {loss} {int(dist)}\n'
        print(f'experiment {loss} {int(dist)}')
        for run in range(n_runs):
            name = f'{dist:.1f}_{loss}_{run}'
            # fp = os.path.join(log_root, name, 'results.txt')
            fp = os.path.join(log_root, name, f"o_data.npy")
            o_data = np.load(fp)
            o_data = torch.from_numpy(o_data)
            o_data0, o_data1, o_data2 = o_data.chunk(3, dim=0)
            o_data0 = o_data0[torch.randperm(12800)[:10000]].reshape(-1, 1)
            o_data1 = o_data1[torch.randperm(12800)[:10000]].reshape(-1, 1)
            o_data2 = o_data2[torch.randperm(12800)[:10000]].reshape(-1, 1)
            o_data = torch.cat([o_data0, o_data1, o_data2], dim=0)
            o_data = o_data[torch.randperm(30000)[:15000]].reshape(-1, 1)
            o_dict = {'0': o_data0, '1': o_data1, '2': o_data2, 'm': o_data}
            # with open(fp, 'r') as f:
            #     lines = f.readlines()
            # if len(lines) < 25:
            #     st()
            # lines = fix_file(lines)
            for m in methods:
                fp = os.path.join(log_root, name, f"{m}_data.npy")
                m_data = np.load(fp)
                m_data = torch.from_numpy(m_data)
                m_data0, m_data1, m_data2 = m_data.chunk(3, dim=0)
                m_data0 = m_data0.reshape(-1, 1)
                m_data1 = m_data1.reshape(-1, 1)
                m_data2 = m_data2.reshape(-1, 1)
                m_data = m_data.reshape(-1, 1)
                m_data = m_data[torch.randperm(30000)[:15000]].reshape(-1, 1)
                m_dict = {'0': m_data0, '1': m_data1, '2': m_data2, 'm': m_data}
                # mix_rbf_mmd2(m_data, o_data, sigma_list)
                for c in classes:
                    # mean, std = parse_(lines[methods.index(m)*5+classes.index(c)+1])
                    # res[m][c].append(mean)
                    res[m][c].append(mix_rbf_mmd2(m_dict[c], o_dict[c], sigma_list).numpy())
                    # res[m][c].append(mean+std)
                    # res[m][c].append(mean-std)
        for m in methods_valid:
            res_full += f'  model: {m}\n'
            for c in classes:
                a = np.array(res[m][c])
                a.sort()
                a = a[:int(1*len(a))]
                if a.max() > 1000:
                    st()
                res_full += f'    {c}: {a.mean():.4f}, {a.std():.4f}\n'
                res[m][c] = a.mean()
                res_mmd[m][dist][c] = (a.mean(), a.std())
        res_full += '\n\n'
        for c in classes:
            cm = []
            for m in methods_valid:
                cm.append(res[m][c])
            cm_ = copy.deepcopy(cm)
            cm_.sort()
            for i, m in enumerate(methods_valid, 0):
                bf_mmd[m][dist][c] = cm_.index(cm[i]) in [0, 1]
                avg_rank[m].append(cm_.index(cm[i])+1)

# with open('results_full_1d.txt', 'w') as f:
#     f.write(res_full)
print(f'Results for loss {config.loss}')
ars = []
for m in methods_valid:
    ar = np.array(avg_rank[m]).mean()
    ars.append(ar)
    print(f'{len(avg_rank[m])} exp for {m}, AvgRank = {ar:.4f}')
ranking = []
j = np.array(ars).argsort()
for i in j:
    ranking.append(methods_valid[i])
ranking = ' > '.join(ranking)
print(f'=> {ranking}.')

### print table
for m in methods_valid:
    print(f"{m}:")
    this_str = ''
    # means = []
    # stds = []
    for d in distances:
        # means += [res_mmd[m][d][c][0] for c in classes]
        # stds += [res_mmd[m][d][c][1] for c in classes]
        for c in classes:
            this_str += '& '
            # if bf_mmd[m][d][c]:
            #     this_str += ' {\\bf' + f" {res_mmd[m][d][c][0]:.2e} $\pm$ {res_mmd[m][d][c][1]:.2e} " + '} '
            # else:
            #     this_str += f" {res_mmd[m][d][c][0]:.2e} $\pm$ {res_mmd[m][d][c][1]:.2e} "
            if bf_mmd[m][d][c]:
                this_str += ' {\\bf' + f" {res_mmd[m][d][c][0]:.2f} " + '} '
            else:
                this_str += f" {res_mmd[m][d][c][0]:.2f} "
    print(this_str)

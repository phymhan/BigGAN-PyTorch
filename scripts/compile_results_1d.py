import os
import numpy as np
import copy
import argparse
import pdb
st = pdb.set_trace

def parse_(s):
    mean, std = s.strip().split(',')
    return float(mean), float(std)

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
parser.add_argument('--outpath', type=str, default='../figs')
parser.add_argument('--loss', type=str, nargs='+', default=['bce'])
config = parser.parse_args()

if config.os == 'linux':
    log_root = '/media/ligong/Passport/Share/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'
else:
    log_root = '/Users/ligong/Library/Group Containers/G69SCX94XU.duck/Library/Application Support/duck/Volumes/dresden/Active/twin-auxiliary-classifiers-gan/MOG/1D/'

n_runs = 100
valid_ratio = 0.9

methods = ['fc', 'tac', 'ac', 'hybrid', 'projection']
methods_valid = ['projection', 'tac', 'fc', 'hybrid']
classes = ['0', '1', '2', 'm']
distances = [d/2.+1 for d in [0, 2, 4, 6, 8]]
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
            fp = os.path.join(log_root, name, 'results.txt')
            if os.path.exists(fp):
                with open(fp, 'r') as f:
                    lines = f.readlines()
                lines = fix_file(lines)
                for m in methods:
                    for c in classes:
                        mean, std = parse_(lines[methods.index(m)*5+classes.index(c)+1])
                        res[m][c].append(mean)
                        # res[m][c].append(mean+std)
                        # res[m][c].append(mean-std)
        for m in methods_valid:
            res_full += f'  model: {m}\n'
            for c in classes:
                a = np.array(res[m][c])
                a.sort()
                a = a[:int(valid_ratio*len(a))]
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
                this_str += ' {\\bf' + f" {res_mmd[m][d][c][0]:.3f} " + '} '
            else:
                this_str += f" {res_mmd[m][d][c][0]:.3f} "
    print(this_str)

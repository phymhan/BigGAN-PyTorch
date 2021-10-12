import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import argparse
import pdb
st = pdb.set_trace


parser = argparse.ArgumentParser()
parser.add_argument('--os', type=str, default='mac')
parser.add_argument('--config', type=str, default='sep')
parser.add_argument('--outpath', type=str, default='../')
parser.add_argument('--name', type=str, default='iccv_V200_proj')
parser.add_argument('--itr', type=str, default='itr20000')
parser.add_argument('--selected_class_id', type=str, default='../selected_intra_classes_V200.npy')
parser.add_argument('--class_ind', type=int, nargs='+', default=[0,1,2])
config = parser.parse_args()

log_root = '../logs'

# expname = 'iccv_V200_p2mi'
# itrname = 'itr20000'
expname = config.name
itrname = config.itr

fp = os.path.join(log_root, expname, 'embedding')
fr = np.load(os.path.join(fp, f'{itrname}_embed_real.npy'))
ff = np.load(os.path.join(fp, f'{itrname}_embed_fake.npy'))
yr = np.load(os.path.join(fp, f'{itrname}_label_real.npy'))
yf = np.load(os.path.join(fp, f'{itrname}_label_fake.npy'))

ycls = np.load(config.selected_class_id)
print(ycls)
yvis = ycls[config.class_ind]
print(yvis)

# Real
idx = yr == yvis[0]
for y_ in yvis[1:]:
    idx = idx | (yr == y_)
fr_ = fr[idx]
yr_ = yr[idx]

# Fake
idx = yf == yvis[0]
for y_ in yvis[1:]:
    idx = idx | (yf == y_)
ff_ = ff[idx]
yf_ = yf[idx]

fall = np.concatenate([fr_, ff_], 0)
yall = np.concatenate([yr_, yf_], 0)

x = TSNE(n_components=2).fit_transform(fall)
df = pd.DataFrame()
df['t-SNE axis-0'] = x[:,0]
df['t-SNE axis-1'] = x[:,1]
df['class'] = [f"+{s}" for s in yr_] + [f"-{s}" for s in yf_]
sns.set()
sns.set_context('paper')
fig = plt.figure(figsize=(5,5))
ax = sns.scatterplot(x='t-SNE axis-0', y='t-SNE axis-1', hue='class', palette=sns.hls_palette(2*len(yvis), h=0.5), data=df, alpha=0.85)
ax.axis('equal')
plt.tight_layout()
fig.savefig(os.path.join(config.outpath, f"{expname}_{itrname}_{'-'.join(map(str, yvis))}_tsne.pdf"))
# plt.show()

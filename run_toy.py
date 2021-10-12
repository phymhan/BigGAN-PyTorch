import os
import shutil
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import random
from utils_sampling import metrics_diversity, metrics_distribution, metrics_distance, metrics_recovered
import pdb
st = pdb.set_trace
# device = "cpu"


""" Define Networks
"""
class G_guassian(nn.Module):
    def __init__(self, nz, n_classes=25, onehot=False, bn=False):
        super(G_guassian, self).__init__()
        self.onehot = onehot
        self.n_classes = n_classes
        if onehot:
            embed_dim = n_classes
        else:
            embed_dim = nz
            self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=nz)
        ngf = 16
        if bn:
            self.decode = nn.Sequential(
                nn.Linear(nz + embed_dim, ngf),  # concat z and emb(y)
                nn.BatchNorm1d(ngf),
                nn.ReLU(),
                nn.Linear(ngf, ngf),
                nn.BatchNorm1d(ngf),
                nn.ReLU(),
                nn.Linear(ngf, ngf),
                nn.BatchNorm1d(ngf),
                nn.ReLU(),
                nn.Linear(ngf, 2),
            )
        else:
            self.decode = nn.Sequential(
                nn.Linear(nz + embed_dim, ngf),  # concat z and emb(y)
                nn.ReLU(),
                nn.Linear(ngf, ngf),
                nn.ReLU(),
                nn.Linear(ngf, ngf),
                nn.ReLU(),
                nn.Linear(ngf, 2),
            )
        #self.__initialize_weights()

    def forward(self, z, label, output=None):
        if self.onehot:
            y = torch.zeros(z.size(0), self.n_classes).to(device).scatter_(1, label.view(-1, 1), 1)
            input = torch.cat([z, y], dim=1)
        else:
            input = torch.cat([z, self.embed(label)], dim=1)
        output = self.decode(input)
        return output

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class D_guassian(nn.Module):
    def __init__(self, n_classes=25, AC=False, TAC=False, proj=True, naive=False, bn=False):
        super(D_guassian, self).__init__()
        self.AC = AC
        self.TAC = TAC
        self.proj = proj
        self.naive = naive
        ndf = 16
        if bn:
            self.encode = nn.Sequential(
                nn.Linear(2, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, ndf),
                nn.BatchNorm1d(ndf),
                nn.LeakyReLU(0.2),
            )
        else:
            self.encode = nn.Sequential(
                nn.Linear(2, ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2),
                nn.Linear(ndf, ndf),
                nn.LeakyReLU(0.2),
            )
        self.gan_linear = nn.Linear(ndf, 1)
        if self.AC:
            self.ac_linear = nn.Linear(ndf, n_classes)
        if self.TAC:
            self.tac_linear = nn.Linear(ndf, n_classes)
        if self.proj:
            self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=ndf)
        if self.naive:
            self.embed = nn.Embedding(num_embeddings=n_classes, embedding_dim=ndf)
            self.gan_linear2 = nn.Linear(ndf, 1)  # for uncond
            self.ac_linear = nn.Linear(ndf, n_classes)
            self.tac_linear = nn.Linear(ndf, n_classes)

        #self.__initialize_weights()

    def forward(self, input, y=None):
        x = self.encode(input)
        x = x.view(input.shape[0], -1)  # phi(x), image embedding
        out = self.gan_linear(x)  # if proj, psi(phi(x)); if (t)ac, uncond
        if self.proj:
            out += torch.sum(self.embed(y) * x, dim=1, keepdim=True)

        ac = None
        tac = None
        if self.AC:
            ac = self.ac_linear(x)
        if self.TAC:
            tac = self.tac_linear(x)
        
        out2 = None  # output of proj when naive is True
        if self.naive:
            out2 = self.gan_linear2(x)
            out2 += torch.sum(self.embed(y) * x, dim=1, keepdim=True)
            ac = self.ac_linear(x)
            tac = self.tac_linear(x)

        return out, ac, tac, out2

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


""" Helper Function and Losses
"""
def sample_data(loader):
    # image and label pair
    while True:
        for batch, label in loader:
            yield batch, label

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# GAN loss with log trick
def loss_bce_dis_real(real_pred):
    target = torch.tensor(1.).to(device).expand_as(real_pred)
    loss = F.binary_cross_entropy_with_logits(real_pred, target)
    return loss

def loss_bce_dis_fake(fake_pred):
    target = torch.tensor(0.).to(device).expand_as(fake_pred)
    loss = F.binary_cross_entropy_with_logits(fake_pred, target)
    return loss

def loss_bce_gen(fake_pred):
    target_real = torch.tensor(1.).to(device).expand_as(fake_pred)
    loss = F.binary_cross_entropy_with_logits(fake_pred, target_real)
    return loss

# Vanilla GAN loss
def loss_vanilla_dis_real(real_pred):
    loss = F.softplus(-real_pred).mean()
    return loss

def loss_vanilla_dis_fake(fake_pred):
    loss = F.softplus(fake_pred).mean()
    return loss

def loss_vanilla_gen(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


""" Training Loop
"""
def train(loader, n_iter, n_step_d, nz, n_classes, G, D, optg, optd, loss_type='proj', gan_loss='vanilla', device='cpu'):
    batch_size = 100
    if gan_loss == 'bce':
        loss_dis_real = loss_bce_dis_real
        loss_dis_fake = loss_bce_dis_fake
        loss_gen = loss_bce_gen
    elif gan_loss == 'vanilla':
        loss_dis_real = loss_vanilla_dis_real
        loss_dis_fake = loss_vanilla_dis_fake
        loss_gen = loss_vanilla_gen
    else:
        raise NotImplementedError

    for i in tqdm(range(n_iter)):
        #####D step
        for _ in range(n_step_d):
            data, label = next(loader)
            data, label = data.to(device), label.to(device)
            d_real, ac_real, tac_real, pd_real = D(data, label)
            z = torch.randn(batch_size, nz).to(device)
            fake_label = torch.LongTensor(batch_size).random_(n_classes).to(device)
            fake_data = G(z, label=fake_label)
            
            d_fake, ac_fake, tac_fake, pd_fake = D(fake_data.detach(), fake_label)
            
            if loss_type == 'p2':
                d_real = d_real + ac_real[range(batch_size), label] - tac_real[range(batch_size), label]
                d_fake = d_fake + ac_fake[range(batch_size), fake_label] - tac_fake[range(batch_size), fake_label]

            D_loss = loss_dis_real(d_real) + loss_dis_fake(d_fake)
            if loss_type in ['ac', 'tac', 'fc', 'p2']:
                D_loss += F.cross_entropy(ac_real, label)
            if loss_type in ['ac', 'tac']:
                D_loss += F.cross_entropy(ac_fake, fake_label)
            if loss_type in ['tac', 'fc', 'p2']:
                D_loss += F.cross_entropy(tac_fake, fake_label)
            if loss_type == 'naive':
                D_loss += loss_dis_real(pd_real) + loss_dis_fake(pd_fake)
                D_loss += F.cross_entropy(ac_real, label) + F.cross_entropy(tac_fake, fake_label)

            optd.zero_grad()
            D_loss.backward()
            optd.step()

        #####G step
        z = torch.randn(batch_size, nz).to(device)
        fake_label = torch.LongTensor(batch_size).random_(n_classes).to(device)
        fake_data = G(z, label=fake_label)
        d_fake, ac_fake, tac_fake, pd_fake = D(fake_data, fake_label)

        if loss_type == 'p2':
            d_fake = d_fake + ac_fake[range(batch_size), fake_label] - tac_fake[range(batch_size), fake_label]
        G_loss = loss_gen(d_fake)

        if loss_type in ['ac', 'tac', 'fc']:
            G_loss += F.cross_entropy(ac_fake, fake_label)
        if loss_type in ['tac', 'fc']:
            G_loss -= F.cross_entropy(tac_fake, fake_label)
        if loss_type == 'naive':
            G_loss += loss_gen(pd_fake)
            G_loss += F.cross_entropy(ac_fake, fake_label) - F.cross_entropy(tac_fake, fake_label)

        optg.zero_grad()
        G_loss.backward()
        optg.step()


""" Data
"""
def generate_real_data(args, dataset='ring', n_classes=8, gauss_std=0.02, seed=0):
    seed_everything(seed)
    device = args.device
    n_sqrt = int(n_classes ** 0.5)
    n_sample_per_class = args.n_sample_per_class

    label_count = 0
    real_data = []
    real_label = []
    centeroids = []
    if dataset == 'grid':
        # half_len = (n_sqrt-1)/2
        half_len = 1
        for x_ in np.linspace(-half_len, half_len, n_sqrt):
            for y_ in np.linspace(half_len, -half_len, n_sqrt):
                pos = torch.tensor([x_, y_]).to(device).unsqueeze(0)
                z_ = torch.randn(n_sample_per_class, 2).to(device) * gauss_std + pos
                # z_ = torch.clamp(z_, min=-half_len, max=half_len)
                label_ = torch.ones(n_sample_per_class).long().to(device) * label_count
                real_data.append(z_.float())
                real_label.append(label_)
                centeroids.append(pos)
                label_count += 1
    elif dataset == 'ring':
        radius = 1
        delta = 2 * np.pi / n_classes
        for i in range(n_classes):
            angle = i * delta
            pos = torch.tensor([radius * np.cos(angle), radius * np.sin(angle)]).to(device).unsqueeze(0)
            z_ = torch.randn(n_sample_per_class, 2).to(device) * gauss_std + pos
            label_ = torch.ones(n_sample_per_class).long().to(device) * label_count
            real_data.append(z_.float())
            real_label.append(label_)
            centeroids.append(pos)
            label_count += 1
    else:
        raise NotImplementedError

    real_data = torch.cat(real_data, 0)
    real_label = torch.cat(real_label, 0)
    centeroids = torch.cat(centeroids, 0)
    train_dset = torch.utils.data.TensorDataset(real_data, real_label)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=100, shuffle=True)
    return real_data, real_label, centeroids, train_dset, train_loader


""" Run Experiment
"""
def run_experiment(args, train_loader, n_classes, nz, config, device, seed=0):
    seed_everything(seed)
    n_step_d = args.n_step_d
    n_iter = args.n_epochs * n_classes * args.n_sample_per_class // n_step_d
    bn = False
    gan_loss = args.gan_loss
    naive = False
    if config == 'proj':
        AC = False
        TAC = False
        proj = True
        loss_type = 'proj'
    elif config == 'fc':
        AC = True
        TAC = True
        proj = False
        loss_type = 'fc'
    elif config == 'ac':
        AC = True
        TAC = False
        proj = False
        loss_type = 'ac'
    elif config == 'tac':
        AC = True
        TAC = True
        proj = False
        loss_type = 'tac'
    elif config == 'p2':
        AC = True
        TAC = True
        proj = False
        loss_type = 'p2'
    elif config == 'naive':
        AC = False
        TAC = False
        proj = False
        naive = True
        loss_type = 'naive'
    else:
        raise NotImplementedError

    #####Bulding network
    G = G_guassian(nz=nz, n_classes=n_classes, onehot=False, bn=bn).to(device)
    D = D_guassian(n_classes=n_classes, AC=AC, TAC=TAC, proj=proj, naive=naive, bn=bn).to(device)

    optg = optim.Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    optd = optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))

    #####Train network
    loader = sample_data(train_loader)
    train(loader, n_iter, n_step_d, nz, n_classes, G, D, optg, optd, loss_type=loss_type, gan_loss=gan_loss, device=device)
    # print('Training finished.')

    return G, D


""" Evaluate
"""
@torch.no_grad()
def evaluate(args, data, label, centeroids, G, D, n_classes, nz, name, out_root, device):
    print(f"evaluate {name}")
    gauss_std = args.std
    threshold = gauss_std * 4
    G.eval()
    # n_sample_g = n_classes * args.n_sample_per_class
    n_sample_g = n_classes * 500
    zg = torch.randn(n_sample_g, nz).to(device)
    yg = torch.LongTensor(n_sample_g).random_(n_classes).to(device)
    data_g = G(z=zg, label=yg).cpu()

    real_batch = data.cpu().numpy()
    # real_label = label.cpu().numpy()
    model_batch = data_g.cpu().numpy()
    model_label = yg.cpu().numpy()
    centers = centeroids.cpu().numpy()
    res_dict = {}

    # rev-kl: kl_div(freq_model, freq_real)
    revkl = metrics_diversity(real_batch, model_batch, centers, threshold)
    res_dict['revkl'] = revkl

    # kl: kl_div(freq_real, freq_model)
    kld = metrics_diversity(model_batch, real_batch, centers, threshold)
    res_dict['kl'] = kld

    # jsd
    jsd = metrics_distribution(real_batch, model_batch, centers, threshold)
    res_dict['js'] = jsd

    # Good
    mean_dist, good_rate = metrics_distance(model_batch, centers, threshold)
    res_dict['mean_dist'] = mean_dist
    res_dict['good_rate'] = good_rate

    # recovered
    rec = metrics_recovered(model_batch, model_label, centers, threshold, thres_good=0.9)
    res_dict['recovered'] = rec

    print(f"{res_dict}")

    res_filename = f"{name}.npz"
    np.savez(os.path.join(out_root, res_filename), **res_dict)

    # plot fake data
    f, ax = plt.subplots(figsize=(6, 6))
    df2 = pd.DataFrame()
    df2['x'] = data_g[:,0].cpu().numpy()
    df2['y'] = data_g[:,1].cpu().numpy()
    df2['label'] = yg.cpu().numpy() + 1
    sns.scatterplot(data=df2, x="x", y="y", hue="label")
    plt.gcf().savefig(os.path.join(out_root, f"{name}_fake.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='ring')
    parser.add_argument("--n_classes", type=int, nargs='*', default=[24])
    parser.add_argument("--n_sample_per_class", type=int, default=100)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--n_step_d", type=int, default=2)
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--std", type=float, default=0.02)
    parser.add_argument("--config", type=str, default='proj')
    parser.add_argument("--gan_loss", type=str, default='vanilla')
    parser.add_argument("--out_root", type=str, default='MOG')
    parser.add_argument("--device", type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    configs = ['ac', 'tac', 'fc', 'naive', 'p2', 'proj']

    for n_classes in args.n_classes:
        save_path = os.path.join(args.out_root, f"{args.dataset}-{n_classes}")
        if os.path.exists(save_path):
            pass
        else:
            os.makedirs(save_path)
        
        # generate real data
        data, label, centeroids, _, loader = generate_real_data(
            args,
            dataset=args.dataset,
            n_classes=n_classes,
            gauss_std=args.std,
            seed=n_classes,
        )

        # plot real data
        f, ax = plt.subplots(figsize=(6, 6))
        df1 = pd.DataFrame()
        df1['x'] = data[:,0].cpu().numpy()
        df1['y'] = data[:,1].cpu().numpy()
        df1['label'] = label.cpu().numpy() + 1
        sns.scatterplot(data=df1, x="x", y="y", hue="label")
        plt.gcf().savefig(os.path.join(save_path, f"{args.dataset}-{n_classes}_real.png"))

        nz = args.nz
        for config in configs:
            for run_id in range(args.n_runs):
                name = f"{args.dataset}-{n_classes}_nz={nz}_{config}_run={run_id}"
                G, D = run_experiment(args, loader, n_classes, nz, config, device, seed=run_id)
                evaluate(args, data, label, centeroids, G, D, n_classes, nz, name, save_path, device)

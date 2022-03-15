import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils

import models
from dataset import Ring8, Grid25
from utils.general_utils import parse_config


class Trainer:
    def __init__(self, config_path: str):
        self.config, self.device, self.log_root = parse_config(config_path)
        if not os.path.exists(os.path.join(self.log_root, 'samples')):
            os.makedirs(os.path.join(self.log_root, 'samples'))
        self.dataset, self.dataloader, self.data_dim = self._get_data()
        (self.G, self.D, self.R, self.optimizerG, self.optimizerD, self.optimizerR,
         self.BCEWithLogits, self.MSE) = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_z = torch.randn((1000, self.config['z_dim']), device=self.device)

    def _get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'ring8':
            dataset = Ring8()
            data_dim = 2
        elif self.config['dataset'] == 'grid25':
            dataset = Grid25()
            data_dim = 2
        elif self.config['dataset'] == 'mnist':
            transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            dataset = dset.MNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            data_dim = 28 * 28
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        return dataset, dataloader, data_dim

    def _prepare_training(self):
        print('==> Preparing training...')
        G = models.Generator(self.config['z_dim'], self.data_dim, ngfs=self.config['ngfs'])
        D = models.Discriminator(self.config['z_dim'], self.data_dim, ndfs=self.config['ndfs'])
        R = models.Reconstructor(self.config['z_dim'], self.data_dim, self.config['nrfs'])
        G.to(device=self.device)
        D.to(device=self.device)
        R.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        optimizerD = optim.Adam(D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        optimizerR = optim.Adam(R.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        BCEWithLogits = nn.BCEWithLogitsLoss()
        MSE = nn.MSELoss()
        return G, D, R, optimizerG, optimizerD, optimizerR, BCEWithLogits, MSE

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.R.load_state_dict(ckpt['R'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)
        self.R.to(device=self.device)

    def save_model(self, model_path):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict(), 'R': self.R.state_dict()}, model_path)

    def train(self):
        print('==> Training...')
        sample_paths = []
        for ep in range(self.config['epochs']):
            self.train_one_epoch(ep)

            if self.config['sample_per_epochs'] and (ep + 1) % self.config['sample_per_epochs'] == 0:
                self.sample_generator(ep, os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))
                sample_paths.append(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.generate_gif(sample_paths, os.path.join(self.log_root, f'samples.gif'))
        self.writer.close()

    def train_one_epoch(self, ep):
        self.G.train()
        self.D.train()
        with tqdm(self.dataloader, desc=f'Epoch {ep}', ncols=120) as pbar:
            for it, X in enumerate(pbar):
                X = X[0] if isinstance(X, list) else X
                X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

                # --------- train discriminator --------- #
                z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                fakeX = self.G(z).detach()
                rec_X = self.R(X).detach()
                d_realz_fakex = self.D(fakeX, z)
                d_fakez_realx = self.D(X, rec_X)
                lossD = (self.BCEWithLogits(d_realz_fakex, torch.ones_like(d_realz_fakex)) +
                         self.BCEWithLogits(d_fakez_realx, torch.zeros_like(d_fakez_realx))) / 2
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

                # --------- train generator --------- #
                if (it + 1) % self.config['d_iters'] == 0:
                    z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                    fakeX = self.G(z)
                    rec_fakeX = self.R(fakeX)
                    d_realz_fakex = self.D(fakeX, z)
                    rec_error = self.MSE(rec_fakeX, z)
                    lossGR = d_realz_fakex.mean() + rec_error.mean()
                    self.optimizerG.zero_grad()
                    self.optimizerR.zero_grad()
                    lossGR.backward()
                    self.optimizerG.step()
                    self.optimizerR.step()
                    self.writer.add_scalar('GR/loss', lossGR.item(), it + ep * len(self.dataloader))

    @torch.no_grad()
    def sample_generator(self, ep, savepath):
        self.G.eval()
        fig, ax = plt.subplots(1, 1)
        if self.config['dataset'] in ['ring8', 'grid25']:
            X = self.G(self.sample_z).cpu()
            realX = torch.stack([d for d in self.dataset], dim=0)
            realX = self.dataset.scaler.inverse_transform(realX)
            X = self.dataset.scaler.inverse_transform(X)
            ax.scatter(realX[:, 0], realX[:, 1], c='green', s=1, alpha=0.5)
            ax.scatter(X[:, 0], X[:, 1], c='blue', s=1)
            ax.axis('scaled')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_title(f'Epoch {ep}')
        elif self.config['dataset'] == 'mnist':
            X = self.G(self.sample_z[:64, :]).cpu()
            X = X.view(-1, 1, 28, 28)
            X = torchvision.utils.make_grid(X, normalize=True, value_range=(-1, 1))
            ax.imshow(torch.permute(X, [1, 2, 0]))
            ax.set_axis_off()
            ax.set_title(f'Epoch {ep}')
        else:
            raise ValueError
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils

import models
from utils.general_utils import parse_config


class Trainer:
    def __init__(self, config_path: str):
        self.config, self.device, self.log_root = parse_config(config_path)
        if not os.path.exists(os.path.join(self.log_root, 'samples')):
            os.makedirs(os.path.join(self.log_root, 'samples'))
        self.dataset, self.dataloader, self.c_dim, self.data_dim = self._get_data()
        self.G, self.D, self.optimizerG, self.optimizerD, self.BCE = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_z = torch.randn((50, self.config['z_dim']), device=self.device)
        self.sample_c = F.one_hot(torch.arange(0, self.c_dim, device=self.device).repeat(5, ), num_classes=self.c_dim)

    def _get_data(self):
        print('==> Getting data...')
        transforms = T.Compose([T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
        dataset = dset.MNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
        data_dim = 28 * 28
        c_dim = 10
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        return dataset, dataloader, c_dim, data_dim

    def _prepare_training(self):
        print('==> Preparing training...')
        G = models.Generator(self.config['z_dim'], self.c_dim, self.data_dim)
        D = models.Discriminator(self.data_dim, self.c_dim)
        G.to(device=self.device)
        D.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['optimizer']['adam']['lr'])
        optimizerD = optim.Adam(D.parameters(), lr=self.config['optimizer']['adam']['lr'])
        BCE = nn.BCELoss()
        return G, D, optimizerG, optimizerD, BCE

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def save_model(self, model_path):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict()}, model_path)

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
            for it, (X, y) in enumerate(pbar):
                X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)
                y = F.one_hot(y, num_classes=self.c_dim).to(device=self.device)

                # --------- train discriminator --------- #
                # min -[log(D(x|y)) + log(1-D(G(z|y)))]
                z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                fake = self.G(z, y).detach()
                realscore, fakescore = self.D(X, y), self.D(fake, y)
                lossD = (self.BCE(realscore, torch.ones_like(realscore)) +
                         self.BCE(fakescore, torch.zeros_like(fakescore))) / 2
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

                # --------- train generator --------- #
                # max -log(1-D(G(z|y))) => min -log(D(G(z|y)))
                if (it + 1) % self.config['d_iters'] == 0:
                    z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                    fake = self.G(z, y)
                    fakescore = self.D(fake, y)
                    lossG = self.BCE(fakescore, torch.ones_like(fakescore))
                    self.optimizerG.zero_grad()
                    lossG.backward()
                    self.optimizerG.step()
                    self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))

    @torch.no_grad()
    def sample_generator(self, ep, savepath):
        self.G.eval()
        X = self.G(self.sample_z, self.sample_c).cpu()
        X = X.view(-1, 1, 28, 28)
        X = torchvision.utils.make_grid(X, nrow=10, normalize=True, value_range=(-1, 1))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(torch.permute(X, [1, 2, 0]))
        ax.set_axis_off()
        ax.set_title(f'Epoch {ep}')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)

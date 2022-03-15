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
        self.dataset, self.dataloader, self.img_channels, self.n_classes = self._get_data()
        self.G, self.D, self.optimizerG, self.optimizerD, self.BCEWithLogits, self.CrossEntropy = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_z = torch.randn((5 * self.n_classes, self.config['z_dim'], 1, 1), device=self.device)
        self.sample_c = F.one_hot(torch.arange(0, self.n_classes, device=self.device).repeat(5, ), num_classes=self.n_classes)
        self.sample_c = self.sample_c.view(5 * self.n_classes, -1, 1, 1)

    def _get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'mnist':
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            dataset = dset.MNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            img_channels = 1
            n_classes = 10
        elif self.config['dataset'] == 'fashion_mnist':
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            dataset = dset.FashionMNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            img_channels = 1
            n_classes = 10
        elif self.config['dataset'] == 'cifar10':
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            dataset = dset.CIFAR10(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            img_channels = 3
            n_classes = 10
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        return dataset, dataloader, img_channels, n_classes

    def _prepare_training(self):
        print('==> Preparing training...')
        G = models.Generator(self.config['z_dim'], self.n_classes, self.img_channels)
        D = models.Discriminator(self.n_classes, self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        optimizerD = optim.Adam(D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        BCEWithLogits = nn.BCEWithLogitsLoss()
        CrossEntropy = nn.CrossEntropyLoss()
        return G, D, optimizerG, optimizerD, BCEWithLogits, CrossEntropy

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
                X = X.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.long)
                y_onehot = F.one_hot(y, num_classes=self.n_classes).view(y.shape[0], -1, 1, 1).to(device=self.device)

                # --------- train discriminator --------- #
                z = torch.randn((X.shape[0], self.config['z_dim'], 1, 1), device=self.device)
                fake = self.G(z, y_onehot).detach()
                realscore, realclass = self.D(X)
                fakescore, fakeclass = self.D(fake)
                loss_score = (self.BCEWithLogits(realscore, torch.ones_like(realscore)) +
                              self.BCEWithLogits(fakescore, torch.zeros_like(fakescore)))
                loss_class = self.CrossEntropy(realclass, y) + self.CrossEntropy(fakeclass, y)
                lossD = loss_score + loss_class
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.writer.add_scalar('D/loss_score', loss_score.item(), it + ep * len(self.dataloader))
                self.writer.add_scalar('D/loss_class', loss_class.item(), it + ep * len(self.dataloader))

                # --------- train generator --------- #
                if (it + 1) % self.config['d_iters'] == 0:
                    z = torch.randn((X.shape[0], self.config['z_dim'], 1, 1), device=self.device)
                    fake = self.G(z, y_onehot)
                    fakescore, fakeclass = self.D(fake)
                    loss_score = self.BCEWithLogits(fakescore, torch.ones_like(fakescore))
                    loss_class = self.CrossEntropy(fakeclass, y)
                    lossG = loss_score + loss_class
                    self.optimizerG.zero_grad()
                    lossG.backward()
                    self.optimizerG.step()
                    self.writer.add_scalar('G/loss_score', loss_score.item(), it + ep * len(self.dataloader))
                    self.writer.add_scalar('G/loss_class', loss_class.item(), it + ep * len(self.dataloader))

    @torch.no_grad()
    def sample_generator(self, ep, savepath):
        self.G.eval()
        X = self.G(self.sample_z, self.sample_c).cpu()
        X = torchvision.utils.make_grid(X, nrow=self.n_classes, normalize=True, value_range=(-1, 1))
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

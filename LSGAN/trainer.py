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
        self.dataset, self.dataloader, self.img_channels, self.data_dim = self._get_data()
        self.G, self.D, self.optimizerG, self.optimizerD, self.MSE = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_z = torch.randn((1000, self.config['z_dim']), device=self.device)
        if self.config['model_arch'] == 'CNN':
            self.sample_z = self.sample_z[:64, :].view(64, self.config['z_dim'], 1, 1)

    def _get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'ring8':
            assert self.config['model_arch'] == 'MLP', 'this dataset only support MLP.'
            dataset = Ring8()
            data_dim = 2
            img_channels = None
        elif self.config['dataset'] == 'grid25':
            assert self.config['model_arch'] == 'MLP', 'this dataset only support MLP.'
            dataset = Grid25()
            data_dim = 2
            img_channels = None
        elif self.config['dataset'] == 'mnist':
            assert self.config['model_arch'] == 'CNN', 'this dataset only support CNN.'
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            dataset = dset.MNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            data_dim = None
            img_channels = 1
        elif self.config['dataset'] == 'fashion_mnist':
            assert self.config['model_arch'] == 'CNN', 'this dataset only support CNN.'
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            dataset = dset.FashionMNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            data_dim = None
            img_channels = 1
        elif self.config['dataset'] == 'celeba':
            assert self.config['model_arch'] == 'CNN', 'this dataset only support CNN.'
            transforms = T.Compose([T.Resize((64, 64)), T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            dataset = dset.CelebA(root=self.config['dataroot'], split='train', transform=transforms, download=False)
            data_dim = None
            img_channels = 3
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        return dataset, dataloader, img_channels, data_dim

    def _prepare_training(self):
        print('==> Preparing training...')
        if self.config['model_arch'] == 'MLP':
            G = models.GeneratorMLP(self.config['z_dim'], self.data_dim)
            D = models.DiscriminatorMLP(self.data_dim)
        elif self.config['model_arch'] == 'CNN':
            G = models.GeneratorCNN(self.config['z_dim'], self.img_channels)
            D = models.DiscriminatorCNN(self.img_channels)
        else:
            raise ValueError('model architecture should be either MLP or CNN.')
        G.to(device=self.device)
        D.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['optimizer']['adam']['lr'])
        optimizerD = optim.Adam(D.parameters(), lr=self.config['optimizer']['adam']['lr'])
        MSE = nn.MSELoss()
        return G, D, optimizerG, optimizerD, MSE

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
            for it, X in enumerate(pbar):
                X = X[0] if isinstance(X, list) else X
                X = X.flatten(start_dim=1) if self.config['model_arch'] == 'MLP' else X
                X = X.to(device=self.device, dtype=torch.float32)

                # --------- train discriminator --------- #
                # min [(D(x)-b)^2 + (D(G(z))-a)^2] / 2
                z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                z = z.view(X.shape[0], self.config['z_dim'], 1, 1) if self.config['model_arch'] == 'CNN' else z
                fake = self.G(z).detach()
                realscore, fakescore = self.D(X), self.D(fake)
                lossD = (self.MSE(realscore, torch.ones_like(realscore) * self.config['b']) +
                         self.MSE(fakescore, torch.ones_like(fakescore) * self.config['a'])) / 2
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

                # --------- train generator --------- #
                # min (D(G(z))-c)^2 / 2
                if (it + 1) % self.config['d_iters'] == 0:
                    z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
                    z = z.view(X.shape[0], self.config['z_dim'], 1, 1) if self.config['model_arch'] == 'CNN' else z
                    fake = self.G(z)
                    fakescore = self.D(fake)
                    lossG = self.MSE(fakescore, torch.ones_like(fakescore) * self.config['c']) / 2
                    self.optimizerG.zero_grad()
                    lossG.backward()
                    self.optimizerG.step()
                    self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))

    @torch.no_grad()
    def sample_generator(self, ep, savepath):
        self.G.eval()
        X = self.G(self.sample_z).cpu()
        fig, ax = plt.subplots(1, 1)
        if self.config['dataset'] in ['ring8', 'grid25']:
            realX = torch.stack([d for d in self.dataset], dim=0)
            realX = self.dataset.scaler.inverse_transform(realX)
            X = self.dataset.scaler.inverse_transform(X)
            ax.scatter(realX[:, 0], realX[:, 1], c='green', s=1, alpha=0.5)
            ax.scatter(X[:, 0], X[:, 1], c='blue', s=1)
            ax.axis('scaled')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_title(f'Epoch {ep}')
        else:  # mnist / fashion_mnist / celeba
            X = torchvision.utils.make_grid(X, normalize=True, value_range=(-1, 1))
            ax.imshow(torch.permute(X, [1, 2, 0]))
            ax.set_axis_off()
            ax.set_title(f'Epoch {ep}')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)

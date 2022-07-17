import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_channels: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, (4, 4), stride=(1, 1), padding=(0, 0)),  # 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1)),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1)),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        if X.ndim == 2:
            X = X.view(-1, X.shape[1], 1, 1)
        return self.gen(X)


class Discriminator(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        self.disc = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 128, (4, 4), stride=(2, 2), padding=1)),  # 32x32
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1)),  # 16x16
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1)),  # 8x8
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(512, 1024, (4, 4), stride=(2, 2), padding=1)),  # 4x4
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(1024, 1, (4, 4), stride=(1, 1), padding=0)),  # 1x1
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        return self.disc(X).squeeze()


class SNGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.img_channels)
        self.D = Discriminator(self.img_channels)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        assert X.shape[-2:] == (64, 64), f'SNGAN only supports 64x64 input.'

        X = X.to(device=self.device, dtype=torch.float32)
        BCE = getattr(self, 'BCE')

        # --------- train discriminator --------- #
        # min -[log(D(x)) + log(1-D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        realscore, fakescore = self.D(X), self.D(fake)
        lossD = (BCE(realscore, torch.ones_like(realscore)) +
                 BCE(fakescore, torch.zeros_like(fakescore))) / 2
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z))) => min -log(D(G(z)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z)
            fakescore = self.D(fake)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


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

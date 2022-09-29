import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, data_dim),
            nn.Tanh(),
        )

    def forward(self, X: torch.Tensor):
        return self.gen(X)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, X: torch.Tensor):
        return self.disc(X)

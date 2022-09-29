from typing import List

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, ngfs: List[int]):
        super().__init__()
        self.gen = []
        last_dim = z_dim + 1
        bn = False
        for d in ngfs:
            self.gen.append(nn.Linear(last_dim, d))
            if bn:
                self.gen.append(nn.BatchNorm1d(d))
            self.gen.append(nn.LeakyReLU(0.2))
            last_dim = d; bn = True
        self.gen.append(nn.Linear(last_dim, data_dim))
        self.gen.append(nn.Tanh())
        self.gen = nn.Sequential(*self.gen)
        self.apply(weights_init)

    def forward(self, Z: torch.Tensor):
        noise = torch.randn((Z.shape[0], 1), device=Z.device)
        return self.gen(torch.cat([Z, noise], dim=1))


class Discriminator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, ndfs: List[int]):
        super().__init__()
        self.disc = []
        last_dim = data_dim + z_dim
        for d in ndfs:
            self.disc.append(nn.Linear(last_dim, d))
            self.disc.append(nn.LeakyReLU(0.2))
            last_dim = d
        self.disc.append(nn.Linear(last_dim, 1))
        self.disc = nn.Sequential(*self.disc)
        self.apply(weights_init)

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        return self.disc(torch.cat([X, Z], dim=1))


class Reconstructor(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, nrfs: List[int]):
        super().__init__()
        self.rec = []
        last_dim = data_dim
        for d in nrfs:
            self.rec.append(nn.Linear(last_dim, d))
            self.rec.append(nn.LeakyReLU(0.2))
            last_dim = d
        self.rec.append(nn.Linear(last_dim, z_dim))
        self.rec = nn.Sequential(*self.rec)
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        mus = self.rec(X)  # self.rec only generate mean values of gaussians
        return torch.randn(mus.shape, device=X.device, requires_grad=True) + mus


def _test():
    G = Generator(z_dim=100, data_dim=2, ngfs=[256, 256])
    D = Discriminator(z_dim=100, data_dim=2, ndfs=[256, 256])
    R = Reconstructor(z_dim=100, data_dim=2, nrfs=[256, 256])
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(z, fakeX)
    reconstructedz = R(fakeX)
    print(fakeX.shape)
    print(score.shape)
    print(reconstructedz.shape)


if __name__ == '__main__':
    _test()

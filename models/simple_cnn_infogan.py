from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.init_weights import init_weights


class GeneratorInfoGAN(nn.Module):
    def __init__(
            self,
            z_dim: int,
            dim: int = 128,
            dim_mults: List[int] = (8, 4, 2, 1),
            dim_c_disc: int = 0,
            dim_c_cont: int = 0,
            out_dim: int = 3,
            with_bn: bool = True,
            with_tanh: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN generator for InfoGAN.

        The network is composed of a stack of convolutional layers.
        The first layer outputs a Tensor with a resolution of 4x4.
        Each following layer doubles the resolution using transposed convolution.

        The categorial conditions are integrated into the network by first transforming
        to one-hot vectors and then concatenating with the input on the channel dimension.

        Args:
            z_dim: Dimension of the latent variable.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            dim_c_disc: Dimension of discrete condition variable, i.e. number of classes.
            dim_c_cont: Dimension of continuous condition variable.
            out_dim: Output dimension.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont

        in_dim = z_dim + dim_c_disc + dim_c_cont
        self.first_conv = nn.ConvTranspose2d(in_dim, dim * dim_mults[0], (4, 4), stride=1, padding=0)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if with_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1),
            ))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim * dim_mults[-1], out_dim, (4, 4), stride=2, padding=1),
        )
        self.act = nn.Tanh() if with_tanh else nn.Identity()

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor, c_disc: Tensor = None, c_cont: Tensor = None):
        if X.ndim == 2:
            X = X[:, :, None, None]
        if c_disc is not None:
            assert self.dim_c_disc != 0 and c_disc.shape == (X.shape[0], )
            c_disc = F.one_hot(c_disc, num_classes=self.dim_c_disc)[:, :, None, None]
            X = torch.cat((X, c_disc), dim=1)
        if c_cont is not None:
            assert c_cont.shape == (X.shape[0], self.dim_c_cont)
            c_cont = c_cont[:, :, None, None]
            X = torch.cat((X, c_cont), dim=1)
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        X = self.act(X)
        return X


class DiscriminatorInfoGAN(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            dim: int = 128,
            dim_mults: List[int] = (1, 2, 4, 8),
            dim_c_disc: int = 0,
            dim_c_cont: int = 0,
            with_bn: bool = True,
            init_type: str = 'normal',
    ):
        """ A simple CNN discriminator for InfoGAN.

        The network is composed of a stack of convolutional layers.
        Each layer except the last layer reduces the resolution by half.
        The last layer reduces the resolution from 4x4 to 1x1.

        The last layer has two branches, the first one outputs True/False
        acting like a typical discriminator, while the second one outputs
        predicted class probabilities acting like a classifier.

        Args:
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            dim_c_disc: Dimension of discrete condition variable, i.e. number of classes.
            dim_c_cont: Dimension of continuous condition variable.
            with_bn: Use batch normalization.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont

        self.first_conv = nn.Conv2d(in_dim, dim * dim_mults[0], (4, 4), stride=2, padding=1)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1)
            ))
        self.last = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disc_head = nn.Conv2d(dim * dim_mults[-1], 1, (4, 4), stride=1, padding=0)
        if dim_c_disc != 0:
            self.c_disc_head = nn.Conv2d(dim * dim_mults[-1], dim_c_disc, (4, 4), stride=1, padding=0)
        if dim_c_cont != 0:
            self.c_cont_head = nn.Conv2d(dim * dim_mults[-1], dim_c_cont, (4, 4), stride=1, padding=0)

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last(X)
        disc = self.disc_head(X).squeeze()
        c_disc, c_cont = None, None
        if self.dim_c_disc != 0:
            c_disc = self.c_disc_head(X).squeeze()
        if self.dim_c_cont != 0:
            c_cont = self.c_cont_head(X).squeeze()
        return disc, c_disc, c_cont


def _test_cbn():
    G = GeneratorInfoGAN(100, dim_c_disc=10, dim_c_cont=2)
    z = torch.randn(5, 100)
    c_disc = torch.randint(0, 10, (5, ))
    c_cont = torch.randn((5, 2))
    fakeX = G(z, c_disc, c_cont)
    D = DiscriminatorInfoGAN(dim_c_disc=10, dim_c_cont=2)
    score, out_disc, out_cont = D(fakeX)
    print(fakeX.shape)
    print(score.shape, out_disc.shape, out_cont.shape)


if __name__ == '__main__':
    _test_cbn()

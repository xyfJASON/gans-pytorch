from typing import List

import torch.nn as nn
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm

from models.init_weights import init_weights


class Generator(nn.Module):
    def __init__(
            self,
            z_dim: int,
            dim: int = 128,
            dim_mults: List[int] = (8, 4, 2, 1),
            out_dim: int = 3,
            with_bn: bool = True,
            with_tanh: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN generator.

        The network is composed of a stack of convolutional layers.
        The first layer outputs a Tensor with a resolution of 4x4.
        Each following layer doubles the resolution using transposed convolution.

        Args:
            z_dim: Dimension of the latent variable.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            out_dim: Output dimension.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.first_conv = nn.ConvTranspose2d(z_dim, dim * dim_mults[0], (4, 4), stride=1, padding=0)
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

    def forward(self, X: Tensor):
        if X.ndim == 2:
            X = X.view(-1, X.shape[1], 1, 1)
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        X = self.act(X)
        return X


class Discriminator(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            dim: int = 128,
            dim_mults: List[int] = (1, 2, 4, 8),
            with_bn: bool = True,
            with_sn: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN discriminator.

        The network is composed of a stack of convolutional layers.
        Each layer except the last layer reduces the resolution by half.
        The last layer reduces the resolution from 4x4 to 1x1.

        Args:
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            with_sn: Use spectral normalization on convolutional layers.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.with_sn = with_sn

        self.first_conv = self.get_conv2d(in_dim, dim * dim_mults[0], (4, 4), stride=2, padding=1)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                self.get_conv2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1)
            ))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            self.get_conv2d(dim * dim_mults[-1], 1, (4, 4), stride=1, padding=0),
        )

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        return X.squeeze()

    def get_conv2d(self, *args, **kwargs):
        if self.with_sn:
            return spectral_norm(nn.Conv2d(*args, **kwargs))
        else:
            return nn.Conv2d(*args, **kwargs)


def _test():
    import torch
    G = Generator(100)
    D = Discriminator()
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(score.shape)


def _test_sn():
    D = Discriminator(with_sn=False)
    print(D)
    D = Discriminator(with_sn=True)
    print(D)


if __name__ == '__main__':
    _test()
    _test_sn()

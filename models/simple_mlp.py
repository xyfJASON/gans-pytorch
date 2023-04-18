from typing import List

import torch.nn as nn
from torch import Tensor

from models.init_weights import init_weights


class Generator(nn.Module):
    def __init__(
            self,
            z_dim: int,
            out_dim: int,
            dim: int = 256,
            dim_mults: List[int] = (1, 1, 1),
            with_bn: bool = False,
            with_tanh: bool = False,
            init_type: str = None,
    ):
        """ A simple MLP generator.

        Args:
            z_dim: Dimension of the latent variable.
            out_dim: Output dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.first_layer = nn.Linear(z_dim, dim * dim_mults[0])
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(dim * dim_mults[i]) if with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(dim * dim_mults[i], dim * dim_mults[i+1]),
            ))
        self.last_layer = nn.Sequential(
            nn.BatchNorm1d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * dim_mults[-1], out_dim),
        )
        self.act = nn.Tanh() if with_tanh else nn.Identity()

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_layer(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_layer(X)
        X = self.act(X)
        return X


class Discriminator(nn.Module):
    def __init__(
            self,
            in_dim: int,
            dim: int = 256,
            dim_mults: List[int] = (1, 1, 1),
            with_bn: bool = False,
            init_type: str = None,
    ):
        """ A simple MLP discriminator.

        Args:
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.first_layer = nn.Linear(in_dim, dim)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm1d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(dim * dim_mults[i], dim * dim_mults[i+1]),
            ))
        self.last_layer = nn.Sequential(
            nn.BatchNorm1d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim * dim_mults[-1], 1),
        )

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        X = self.first_layer(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_layer(X)
        return X


def _test():
    import torch
    G = Generator(100, 1000)
    D = Discriminator(1000)
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(score.shape)


if __name__ == '__main__':
    _test()

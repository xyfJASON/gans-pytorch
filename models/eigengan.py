from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.init_weights import init_weights


class SubspaceLayer(nn.Module):
    def __init__(self, n_basis: int, dim: int):
        super().__init__()
        self.U = nn.Parameter(torch.empty(n_basis, dim))
        self.L = nn.Parameter(torch.ones(n_basis))
        self.mu = nn.Parameter(torch.zeros(dim))
        nn.init.orthogonal_(self.U)

    def forward(self, z: Tensor):
        """
        Args:
            z: [bs, n_basis]
        """
        z = z * self.L[None, :]             # [bs, n_basis]
        h = z @ self.U + self.mu[None, :]   # [bs, dim]
        ortho_reg = torch.mean((self.U @ self.U.T - torch.eye(z.shape[1], device=z.device)) ** 2)
        return h, ortho_reg


class EigenBlock(nn.Module):
    def __init__(self, n_basis: int, in_channels: int, out_channels: int, size: int, with_bn: bool = True):
        super().__init__()
        self.subspace = SubspaceLayer(n_basis, in_channels * size * size)
        self.dconv1 = nn.ConvTranspose2d(in_channels, in_channels, (1, 1), stride=1, padding=0)
        self.dconv2 = nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=2, padding=1, output_padding=1)
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=2, padding=1, output_padding=1),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
        )

    def forward(self, x: Tensor, z: Tensor):
        """
        Args:
            x: [bs, c, h, w]
            z: [bs, n_basis]
        """
        h, ortho_reg = self.subspace(z)
        h = h.reshape(x.shape)                  # [bs, ci, h, w]
        x = self.block1(x + self.dconv1(h))     # [bs, co, 2h, 2w]
        x = self.block2(x + self.dconv2(h))     # [bs, co, 2h, 2w]
        return x, ortho_reg


class Generator(nn.Module):
    def __init__(
            self,
            n_basis: int = 6,
            noise_dim: int = 512,
            out_dim: int = 3,
            dim: int = 16,
            dim_mults: List[int] = (32, 16, 8, 4, 2, 1),
            with_bn: bool = False,
            with_tanh: bool = True,
            init_type: str = 'normal',
    ):
        """The EigenGAN generator.

        Args:
            n_basis: Number of the basis.
            noise_dim: Dimensionality of the input noise.
            out_dim: Output dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.noise_dim = noise_dim

        self.first_layer = nn.Linear(noise_dim, noise_dim * 4 * 4)
        cur_dim = noise_dim
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults)):
            self.layers.append(EigenBlock(
                n_basis=n_basis,
                in_channels=cur_dim,
                out_channels=dim * dim_mults[i],
                size=4 * (2 ** i),
                with_bn=with_bn,
            ))
            cur_dim = dim * dim_mults[i]
        self.last_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(cur_dim, out_dim, (7, 7), stride=1, padding=3),
        )
        self.act = nn.Tanh() if with_tanh else nn.Identity()

        self.apply(init_weights(init_type))

    def forward(self, x: Tensor, z: Tensor, return_ortho_reg: bool = False):
        """
        Args:
            x: [bs, noise_dim]
            z: [bs, n_layer, n_basis]
            return_ortho_reg: return orthonormal regularization
        """
        x = self.first_layer(x)
        x = x.reshape(-1, self.noise_dim, 4, 4)
        ortho_reg = 0.0
        for i, layer in enumerate(self.layers):
            x, o = layer(x, z[:, i, :])
            ortho_reg = ortho_reg + o
        x = self.last_layer(x)
        x = self.act(x)
        if not return_ortho_reg:
            return x
        return x, ortho_reg


class Discriminator(nn.Module):
    def __init__(
            self,
            in_dim: int = 3,
            dim: int = 16,
            dim_mults: List[int] = (2, 4, 8, 16, 32, 32),
            with_bn: bool = False,
            init_type: str = 'normal',
    ):
        """The EigenGAN discriminator.

        Args:
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            init_type: Type of weight initialization.

        """
        super().__init__()

        self.first_conv = nn.Conv2d(in_dim, dim, (7, 7), stride=1, padding=3)
        cur_dim = dim
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults)):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(cur_dim) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(cur_dim, cur_dim, (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(cur_dim) if with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(cur_dim, dim * dim_mults[i], (3, 3), stride=2, padding=1),
            ))
            cur_dim = dim * dim_mults[i]
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(cur_dim) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(cur_dim, cur_dim, (3, 3), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
        )
        self.cls = nn.Sequential(
            nn.Linear(cur_dim * 4 * 4, cur_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cur_dim, 1),
        )

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        X = self.cls(X)
        return X


def _test():
    G = Generator()
    D = Discriminator()
    print(sum([p.numel() for p in G.parameters()]))
    print(sum([p.numel() for p in D.parameters()]))

    z = torch.randn(10, 6, 6)
    x = torch.randn(10, 512)
    x = G(x, z)
    print(x.shape)

    y = D(x)
    print(y.shape)


if __name__ == '__main__':
    _test()

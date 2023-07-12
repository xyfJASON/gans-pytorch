from typing import List

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from models.init_weights import init_weights
from models.modules import ConditionalBatchNorm2d


class GeneratorConditional(nn.Module):
    def __init__(
            self,
            z_dim: int,
            n_classes: int,
            dim: int = 128,
            dim_mults: List[int] = (8, 4, 2, 1),
            out_dim: int = 3,
            with_bn: bool = True,
            with_tanh: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN generator conditioned on class labels.

        The network is composed of a stack of convolutional layers.
        The first layer outputs a Tensor with a resolution of 4x4.
        Each following layer doubles the resolution using transposed convolution.

        The categorial conditions are integrated into the network by first transforming
        to one-hot vectors and then concatenating with the input on the channel dimension.

        Args:
            z_dim: Dimension of the latent variable.
            n_classes: Number of classes.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            out_dim: Output dimension.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.n_classes = n_classes

        self.first_conv = nn.ConvTranspose2d(z_dim + n_classes, dim * dim_mults[0], (4, 4), stride=1, padding=0)
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

    def forward(self, X: Tensor, y: Tensor):
        if X.ndim == 2:
            X = X[:, :, None, None]
        y = F.one_hot(y, num_classes=self.n_classes)[:, :, None, None]
        X = torch.cat((X, y), dim=1)
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last_conv(X)
        X = self.act(X)
        return X


class GeneratorConditionalCBN(nn.Module):
    def __init__(
            self,
            z_dim: int,
            n_classes: int,
            dim: int = 128,
            dim_mults: List[int] = (8, 4, 2, 1),
            out_dim: int = 3,
            with_bn: bool = True,
            with_tanh: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN generator conditioned on class labels.

        The network is composed of a stack of convolutional layers.
        The first layer outputs a Tensor with a resolution of 4x4.
        Each following layer doubles the resolution using transposed convolution.

        The categorial conditions are integrated into the network by conditional
        batch normalization layer.

        Args:
            z_dim: Dimension of the latent variable.
            n_classes: Number of classes.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            out_dim: Output dimension.
            with_bn: Use batch normalization.
            with_tanh: Use nn.Tanh() at last.
            init_type: Type of weight initialization.

        """
        super().__init__()
        assert with_bn is True, f'class `GeneratorConditionalCBN` requires with_bn to be `True`'
        self.n_classes = n_classes

        self.first_conv = nn.ConvTranspose2d(z_dim, dim * dim_mults[0], (4, 4), stride=1, padding=0)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.ModuleList([
                ConditionalBatchNorm2d(n_classes, dim * dim_mults[i]),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1),
            ]))
        self.last_conv = nn.ModuleList([
            ConditionalBatchNorm2d(n_classes, dim * dim_mults[-1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim * dim_mults[-1], out_dim, (4, 4), stride=2, padding=1),
        ])
        self.act = nn.Tanh() if with_tanh else nn.Identity()

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor, y: Tensor):
        if X.ndim == 2:
            X = X[:, :, None, None]
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer[0](X, y)          # ConditionalBatchNorm2d
            X = layer[1](X)             # ReLU
            X = layer[2](X)             # ConvTranspose2d
        X = self.last_conv[0](X, y)     # ConditionalBatchNorm2d
        X = self.last_conv[1](X)        # ReLU
        X = self.last_conv[2](X)        # ConvTranspose2d
        X = self.act(X)
        return X


class DiscriminatorConditional(nn.Module):
    def __init__(
            self,
            n_classes: int,
            in_dim: int = 3,
            dim: int = 128,
            dim_mults: List[int] = (1, 2, 4, 8),
            with_bn: bool = True,
            with_sn: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN discriminator conditioned on class labels.

        The network is composed of a stack of convolutional layers.
        Each layer except the last layer reduces the resolution by half.
        The last layer reduces the resolution from 4x4 to 1x1.

        The categorial conditions are integrated into the network by first transforming
        to one-hot vectors and then concatenating with the input on the channel dimension.

        Args:
            n_classes: Number of classes.
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            with_sn: Use spectral normalization on convolutional layers.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.n_classes = n_classes
        self.with_sn = with_sn

        self.first_conv = self.get_conv2d(in_dim + n_classes, dim * dim_mults[0], (4, 4), stride=2, padding=1)
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

    def forward(self, X: Tensor, y: Tensor):
        y = F.one_hot(y, num_classes=self.n_classes)[:, :, None, None]
        y = y.expand((y.shape[0], y.shape[1], *X.shape[-2:]))
        X = torch.cat((X, y), dim=1)
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


class DiscriminatorConditionalAC(nn.Module):
    def __init__(
            self,
            n_classes: int,
            in_dim: int = 3,
            dim: int = 128,
            dim_mults: List[int] = (1, 2, 4, 8),
            with_bn: bool = True,
            with_sn: bool = False,
            init_type: str = 'normal',
    ):
        """ A simple CNN discriminator conditioned on class labels.

        The network is composed of a stack of convolutional layers.
        Each layer except the last layer reduces the resolution by half.
        The last layer reduces the resolution from 4x4 to 1x1.

        The last layer has two branches, the first one outputs True/False
        acting like a typical discriminator, while the second one outputs
        predicted class probabilities acting like a classifier.

        Args:
            n_classes: Number of classes.
            in_dim: Input dimension.
            dim: Base dimension.
            dim_mults: Multiplies of dimensions.
            with_bn: Use batch normalization.
            with_sn: Use spectral normalization on convolutional layers.
            init_type: Type of weight initialization.

        """
        super().__init__()
        self.n_classes = n_classes
        self.with_sn = with_sn

        self.first_conv = self.get_conv2d(in_dim, dim * dim_mults[0], (4, 4), stride=2, padding=1)
        self.layers = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(dim * dim_mults[i]) if i > 0 and with_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                self.get_conv2d(dim * dim_mults[i], dim * dim_mults[i+1], (4, 4), stride=2, padding=1)
            ))
        self.last = nn.Sequential(
            nn.BatchNorm2d(dim * dim_mults[-1]) if with_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.disc_head = self.get_conv2d(dim * dim_mults[-1], 1, (4, 4), stride=1, padding=0)
        self.cls_head = self.get_conv2d(dim * dim_mults[-1], n_classes, (4, 4), stride=1, padding=0)

        self.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.first_conv(X)
        for layer in self.layers:
            X = layer(X)
        X = self.last(X)
        disc = self.disc_head(X)
        cls = self.cls_head(X)
        return disc.squeeze(), cls.squeeze()

    def get_conv2d(self, *args, **kwargs):
        if self.with_sn:
            return spectral_norm(nn.Conv2d(*args, **kwargs))
        else:
            return nn.Conv2d(*args, **kwargs)


def _test():
    G = GeneratorConditional(100, 10)
    D = DiscriminatorConditional(10)
    z = torch.randn(5, 100)
    y = torch.randint(0, 10, (5, ))
    fakeX = G(z, y)
    score = D(fakeX, y)
    print(fakeX.shape)
    print(score.shape)


def _test_cbn():
    G = GeneratorConditionalCBN(100, 10)
    z = torch.randn(5, 100)
    y = torch.randint(0, 10, (5, ))
    fakeX = G(z, y)
    print(fakeX.shape)


def _test_ac():
    D = DiscriminatorConditionalAC(10)
    X = torch.randn(5, 3, 64, 64)
    score, logits = D(X)
    print(score.shape, logits.shape)


if __name__ == '__main__':
    # _test()
    # _test_cbn()
    _test_ac()

import torch
import torch.nn as nn
from torch import Tensor


class LSGANLoss(nn.Module):
    """ The Least Squares GAN loss.

    Objective of the discriminator:
        min 1/2 * E[(D(x)-b)^2] + 1/2 * E[(D(G(z))-a)^2]

    Objective of the generator:
        min 1/2 * E[(D(G(z))-c)^2]

    References:
        Mao, Xudong, Qing Li, Haoran Xie, Raymond YK Lau, Zhen Wang, and Stephen Paul Smolley. "Least squares
        generative adversarial networks." In Proceedings of the IEEE international conference on computer vision,
        pp. 2794-2802. 2017.

    """
    def __init__(self, discriminator: nn.Module, a: float, b: float, c: float):
        super().__init__()
        self.discriminator = discriminator
        self.a, self.b, self.c = a, b, c
        self.mse = nn.MSELoss()

    def forward_G(self, fake_data: Tensor):
        fake_score = self.discriminator(fake_data)
        return self.mse(fake_score, torch.full_like(fake_score, self.c)) / 2

    def forward_D(self, fake_data: Tensor, real_data: Tensor):
        fake_score = self.discriminator(fake_data.detach())
        real_score = self.discriminator(real_data)
        return (self.mse(real_score, torch.full_like(real_score, self.b)) +
                self.mse(fake_score, torch.full_like(fake_score, self.a))) / 2

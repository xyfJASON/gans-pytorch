import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class HingeLoss(nn.Module):
    """ The hinge loss for GANs.

    Objective of the discriminator:
        max E[min(0, -1+D(x))] + E[min(0, -1-D(G(z)))] =>
        min E[max(0, 1-D(x))] + E[max(0, 1+D(G(z)))]

    Objective of the generator:
        max E[D(G(z))] => min -E[D(G(z))]

    References:
        Lim, Jae Hyun, and Jong Chul Ye. "Geometric gan." arXiv preprint arXiv:1705.02894 (2017).

    """
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator

    def forward_G(self, fake_data: Tensor):
        fake_score = self.discriminator(fake_data)
        return -torch.mean(fake_score)

    def forward_D(self, fake_data: Tensor, real_data: Tensor):
        fake_score = self.discriminator(fake_data.detach())
        real_score = self.discriminator(real_data)
        return torch.mean(F.relu(1 - real_score) + F.relu(1 + fake_score))

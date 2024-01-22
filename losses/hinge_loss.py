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

    Also supports R1 regularization on the discriminator:
        E[||grad(D(x))||^2] where x is real data

    References:
        Lim, Jae Hyun, and Jong Chul Ye. "Geometric gan." arXiv preprint arXiv:1705.02894 (2017).

        Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which training methods for GANs do actually converge?."
        In International conference on machine learning, pp. 3481-3490. PMLR, 2018.

    """
    def __init__(self, discriminator: nn.Module, lambda_r1_reg: float = 0.0):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_r1_reg = lambda_r1_reg

    @staticmethod
    def _discard_extras(x):
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def r1_reg(self, real_data: Tensor, *args):
        real_data.requires_grad_()
        real_score = self._discard_extras(self.discriminator(real_data, *args))
        gradients = torch.autograd.grad(
            outputs=real_score, inputs=real_data,
            grad_outputs=torch.ones_like(real_score),
            create_graph=True, retain_graph=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)
        return gradients.pow(2).sum(1).mean()

    def forward_G(self, fake_data: Tensor):
        fake_score = self.discriminator(fake_data)
        return -torch.mean(fake_score)

    def forward_D(self, fake_data: Tensor, real_data: Tensor):
        fake_score = self.discriminator(fake_data.detach())
        real_score = self.discriminator(real_data)
        loss = torch.mean(F.relu(1 - real_score) + F.relu(1 + fake_score))
        if self.lambda_r1_reg > 0.0:
            loss = loss + self.lambda_r1_reg / 2 * self.r1_reg(real_data)
        return loss

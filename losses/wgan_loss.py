import torch
import torch.nn as nn
from torch import Tensor


class WGANLoss(nn.Module):
    r""" The Wasserstein GAN loss.

    Objective of the discriminator:
        max E[D(x)] - E[D(G(z))] <=> min E[D(G(z))] - E[D(x)]
    if gradient penalty is enabled:
        min E[D(G(z))] - E[D(x)] + lambda * gp

    Objective of the generator:
        max E[D(G(z))] <=> min E[-D(G(z))]

    References:
        Arjovsky, Martin, and Léon Bottou. "Towards principled methods for training generative adversarial networks."
        arXiv preprint arXiv:1701.04862 (2017).

        Arjovsky, Martin, Soumith Chintala, and Léon Bottou. “Wasserstein generative adversarial networks.”
        In International conference on machine learning, pp. 214-223. PMLR, 2017.

        Gulrajani, Ishaan, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C. Courville. "Improved training
        of wasserstein gans." Advances in neural information processing systems 30 (2017).

    """
    def __init__(self, discriminator: nn.Module, lambda_gp: float = 10.0):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, fake_data: Tensor, real_data: Tensor):
        real_data = real_data.view(*fake_data.shape)
        alpha = torch.rand((1, ), device=fake_data.device)
        inter_data = alpha * real_data + (1 - alpha) * fake_data
        inter_data.requires_grad_()
        d_inter_data = self.discriminator(inter_data)
        gradients = torch.autograd.grad(
            outputs=d_inter_data, inputs=inter_data,
            grad_outputs=torch.ones_like(d_inter_data),
            create_graph=True, retain_graph=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)
        return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def forward_G(self, fake_data: Tensor):
        fake_score = self.discriminator(fake_data)
        return -torch.mean(fake_score)

    def forward_D(self, fake_data: Tensor, real_data: Tensor):
        fake_score = self.discriminator(fake_data.detach())
        real_score = self.discriminator(real_data)
        loss = torch.mean(fake_score) - torch.mean(real_score)
        if self.lambda_gp > 0.0:
            loss = loss + self.lambda_gp * self.gradient_penalty(fake_data, real_data)
        return loss

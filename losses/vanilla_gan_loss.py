import torch
import torch.nn as nn
from torch import Tensor


class VanillaGANLoss(nn.Module):
    """ The vanilla GAN loss (-log alternative).

    Objective of the discriminator:
        min -E[log(D(x))] + E[log(1-D(G(z)))]

    Objective of the generator:
        max -E[log(1-D(G(z)))] => min -E[log(D(G(z)))]

    References:
        Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
        and Yoshua Bengio. “Generative adversarial nets.” Advances in neural information processing systems 27 (2014).

    """
    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward_G(self, fake_data: Tensor, *args):
        fake_score = self.discriminator(fake_data, *args)
        return self.bce_with_logits(fake_score, torch.ones_like(fake_score))

    def forward_D(self, fake_data: Tensor, real_data: Tensor, *args):
        fake_score = self.discriminator(fake_data.detach(), *args)
        real_score = self.discriminator(real_data, *args)
        return (self.bce_with_logits(fake_score, torch.zeros_like(fake_score)) +
                self.bce_with_logits(real_score, torch.ones_like(real_score))) / 2

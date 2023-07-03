import torch
import torch.nn as nn
from torch import Tensor


class VanillaGANLoss(nn.Module):
    """ The vanilla GAN loss (-log alternative).

    Objective of the discriminator:
        min -E[log(D(x))] + E[log(1-D(G(z)))]

    Objective of the generator:
        max -E[log(1-D(G(z)))] => min -E[log(D(G(z)))]

    Also supports R1 regularization on the discriminator:
        E[||grad(D(x))||^2] where x is real data

    References:
        Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
        and Yoshua Bengio. "Generative adversarial nets." Advances in neural information processing systems 27 (2014).

        Mescheder, Lars, Andreas Geiger, and Sebastian Nowozin. "Which training methods for GANs do actually converge?."
        In International conference on machine learning, pp. 3481-3490. PMLR, 2018.

    """
    def __init__(self, discriminator: nn.Module, lambda_r1_reg: float = 0.0):
        super().__init__()
        self.discriminator = discriminator
        self.lambda_r1_reg = lambda_r1_reg
        self.bce_with_logits = nn.BCEWithLogitsLoss()

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

    def forward_G(self, fake_data: Tensor, *args):
        fake_score = self._discard_extras(self.discriminator(fake_data, *args))
        return self.bce_with_logits(fake_score, torch.ones_like(fake_score))

    def forward_D(self, fake_data: Tensor, real_data: Tensor, *args):
        fake_score = self._discard_extras(self.discriminator(fake_data.detach(), *args))
        real_score = self._discard_extras(self.discriminator(real_data, *args))
        loss = (self.bce_with_logits(fake_score, torch.zeros_like(fake_score)) +
                self.bce_with_logits(real_score, torch.ones_like(real_score))) / 2
        if self.lambda_r1_reg > 0.0:
            loss = loss + self.lambda_r1_reg / 2 * self.r1_reg(real_data)
        return loss


class VanillaGANWithAuxiliaryClassifierLoss(nn.Module):
    """ The vanilla GAN loss (-log alternative) with auxiliary classifier. """

    def __init__(self, discriminator: nn.Module):
        super().__init__()
        self.discriminator = discriminator
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward_G(self, fake_data: Tensor, labels: Tensor):
        fake_score, fake_logits = self.discriminator(fake_data)
        loss_disc = self.bce_with_logits(fake_score, torch.ones_like(fake_score))
        loss_cls = self.ce(fake_logits, labels)
        return loss_disc, loss_cls

    def forward_D(self, fake_data: Tensor, real_data: Tensor, labels: Tensor):
        fake_score, fake_logits = self.discriminator(fake_data.detach())
        real_score, real_logits = self.discriminator(real_data)
        loss_disc = (self.bce_with_logits(fake_score, torch.zeros_like(fake_score)) +
                     self.bce_with_logits(real_score, torch.ones_like(real_score))) / 2
        loss_cls = (self.ce(fake_logits, labels) + self.ce(real_logits, labels)) / 2
        return loss_disc, loss_cls

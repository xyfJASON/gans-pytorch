import torch
import torch.nn as nn
from torch import Tensor


class VanillaGANLoss(nn.Module):
    """ The vanilla GAN loss (-log alternative).

    Objective of the discriminator:
        min -[log(D(x)) + log(1-D(G(z)))]

    Objective of the generator:
        max -log(1-D(G(z))) => min -log(D(G(z)))

    References:
        Goodfellow, Ian, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville,
        and Yoshua Bengio. “Generative adversarial nets.” Advances in neural information processing systems 27 (2014).

    """
    def __init__(self):
        super().__init__()
        self.bce_with_logits = nn.BCEWithLogitsLoss()

    def forward_G(self, fakescore: Tensor):
        return self.bce_with_logits(fakescore, torch.ones_like(fakescore))

    def forward_D(self, fakescore: Tensor, realscore: Tensor):
        return (self.bce_with_logits(fakescore, torch.zeros_like(fakescore)) +
                self.bce_with_logits(realscore, torch.ones_like(realscore))) / 2

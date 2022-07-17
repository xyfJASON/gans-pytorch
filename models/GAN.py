import torch
import torch.nn as nn
import torch.optim as optim

from train import BaseTrainer


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh(),
        )

    def forward(self, X: torch.Tensor):
        return self.gen(X)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor):
        return self.disc(X)


class GAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.data_dim)
        self.D = Discriminator(self.data_dim)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'])

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)
        BCE = getattr(self, 'BCE')

        # --------- train discriminator --------- #
        # min -[log(D(x)) + log(1-D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        realscore, fakescore = self.D(X), self.D(fake)
        lossD = (BCE(realscore, torch.ones_like(realscore)) +
                 BCE(fakescore, torch.zeros_like(fakescore))) / 2
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z))) => min -log(D(G(z)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z)
            fakescore = self.D(fake)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


def _test():
    G = Generator(100, 1000)
    D = Discriminator(1000)
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(fakeX)
    print(score.shape)
    print(score)


if __name__ == '__main__':
    _test()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from train import BaseTrainer


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, c_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.gen = nn.Sequential(
            nn.Linear(z_dim + c_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, data_dim),
            nn.Tanh(),
        )

    def forward(self, X: torch.Tensor, C: torch.Tensor):
        assert X.shape[1] == self.z_dim
        assert C.shape[1] == self.c_dim
        return self.gen(torch.cat((X, C), dim=1))


class Discriminator(nn.Module):
    def __init__(self, in_dim: int, c_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.c_dim = c_dim
        self.disc = nn.Sequential(
            nn.Linear(in_dim + c_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, X: torch.Tensor, C: torch.Tensor):
        assert X.ndim == 2 and X.shape[1] == self.in_dim
        assert C.ndim == 2 and C.shape[1] == self.c_dim
        return self.disc(torch.cat((X, C), dim=1))


class CGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, conditional=True)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.data_dim, self.n_classes)
        self.D = Discriminator(self.data_dim, self.n_classes)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'])

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)
        y = F.one_hot(y, num_classes=self.n_classes).to(device=self.device)
        BCE = getattr(self, 'BCE')

        # --------- train discriminator --------- #
        # min -[log(D(x|y)) + log(1-D(G(z|y)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z, y).detach()
        realscore, fakescore = self.D(X, y), self.D(fake, y)
        lossD = (BCE(realscore, torch.ones_like(realscore)) +
                 BCE(fakescore, torch.zeros_like(fakescore))) / 2
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z|y))) => min -log(D(G(z|y)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z, y)
            fakescore = self.D(fake, y)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


def _test():
    c = torch.randint(0, 10, (10, ))
    c = F.one_hot(c, num_classes=10)
    G = Generator(100, 10, 1000)
    D = Discriminator(1000, 10)
    z = torch.randn(10, 100)
    fakeX = G(z, c)
    score = D(fakeX, c)
    print(fakeX.shape)
    print(score.shape)


if __name__ == '__main__':
    _test()

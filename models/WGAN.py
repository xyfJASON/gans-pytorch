import torch
import torch.nn as nn
import torch.optim as optim

from train import BaseTrainer


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, data_dim),
            nn.Tanh(),
        )

    def forward(self, X: torch.Tensor):
        return self.gen(X)


class Discriminator(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, X: torch.Tensor):
        return self.disc(X)


class WGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.data_dim)
        self.D = Discriminator(self.data_dim)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.RMSprop(self.G.parameters(), lr=self.config['optimizer']['rmsprop']['lr'])
        self.optimizerD = optim.RMSprop(self.D.parameters(), lr=self.config['optimizer']['rmsprop']['lr'])

    def define_losses(self):
        pass

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # max E[D(x)]-E[D(G(z))] <=> min E[D(G(z))]-E[D(x)]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        lossD = torch.mean(self.D(fake)) - torch.mean(self.D(X))
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        for param in self.D.parameters():
            param.data.clamp_(min=self.config['clip'][0], max=self.config['clip'][1])  # weight clipping

        # --------- train generator --------- #
        # max E[D(G(z))] <=> min E[-D(G(z))]
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z)
            lossG = -torch.mean(self.D(fake))
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))

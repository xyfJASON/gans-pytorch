import torch
import torch.nn as nn
import torch.optim as optim

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class GeneratorMLP(nn.Module):
    def __init__(self, z_dim: int, data_dim: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
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
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        return self.gen(X)


class DiscriminatorMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        return self.disc(X)


class GeneratorCNN(nn.Module):
    def __init__(self, z_dim: int, img_channels: int):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, (4, 4), stride=(1, 1), padding=(0, 0)),  # 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1)),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1)),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        if X.ndim == 2:
            X = X.view(-1, X.shape[1], 1, 1)
        return self.gen(X)


class DiscriminatorCNN(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 128, (4, 4), stride=(2, 2), padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, (4, 4), stride=(2, 2), padding=1),  # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, (4, 4), stride=(1, 1), padding=0),  # 1x1
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        return self.disc(X).squeeze()


class LSGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        if self.config['model_arch'] == 'MLP':
            self.G = GeneratorMLP(self.config['z_dim'], self.data_dim)
            self.D = DiscriminatorMLP(self.data_dim)
        elif self.config['model_arch'] == 'CNN':
            self.G = GeneratorCNN(self.config['z_dim'], self.img_channels)
            self.D = DiscriminatorCNN(self.img_channels)
        else:
            raise ValueError('model architecture should be either MLP or CNN.')
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'])

    def define_losses(self):
        setattr(self, 'MSE', nn.MSELoss())

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1) if self.config['model_arch'] == 'MLP' else X
        X = X.to(device=self.device, dtype=torch.float32)
        MSE = getattr(self, 'MSE')

        # --------- train discriminator --------- #
        # min [(D(x)-b)^2 + (D(G(z))-a)^2] / 2
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        realscore, fakescore = self.D(X), self.D(fake)
        lossD = (MSE(realscore, torch.ones_like(realscore) * self.config['b']) +
                 MSE(fakescore, torch.ones_like(fakescore) * self.config['a'])) / 2
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # min (D(G(z))-c)^2 / 2
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z)
            fakescore = self.D(fake)
            lossG = MSE(fakescore, torch.ones_like(fakescore) * self.config['c']) / 2
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


def _test():
    G = GeneratorMLP(100, 1000)
    D = DiscriminatorMLP(1000)
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX)
    print(score)


if __name__ == '__main__':
    _test()

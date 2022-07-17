import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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
        self.apply(weights_init)

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
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        return self.disc(X)


class WGAN_GP_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.data_dim)
        self.D = Discriminator(self.data_dim)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])

    def define_losses(self):
        pass

    def gradient_penalty(self, realX: torch.Tensor, fakeX: torch.Tensor):
        alpha = torch.rand(1, device=self.device)
        interX = alpha * realX + (1 - alpha) * fakeX
        interX.requires_grad_()
        d_interX = self.D(interX)
        gradients = autograd.grad(outputs=d_interX, inputs=interX,
                                  grad_outputs=torch.ones_like(d_interX),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.flatten(start_dim=1)
        return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min E[D(G(z))] - E[D(x)] + lambda * gp
        self.D.zero_grad()
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        lossD = torch.mean(self.D(fake)) - torch.mean(self.D(X)) + self.config['lambda_gp'] * self.gradient_penalty(X, fake)
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

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


def _test():
    G = Generator(100, 1000)
    D = Discriminator(1000)
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(score.shape)


if __name__ == '__main__':
    _test()

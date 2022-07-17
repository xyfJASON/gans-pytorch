import torch
import torch.nn as nn
import torch.optim as optim

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, ngfs: list[int]):
        super().__init__()
        self.gen = []
        last_dim = z_dim + 1
        bn = False
        for d in ngfs:
            self.gen.append(nn.Linear(last_dim, d))
            if bn:
                self.gen.append(nn.BatchNorm1d(d))
            self.gen.append(nn.LeakyReLU(0.2))
            last_dim = d; bn = True
        self.gen.append(nn.Linear(last_dim, data_dim))
        self.gen.append(nn.Tanh())
        self.gen = nn.Sequential(*self.gen)
        self.apply(weights_init)

    def forward(self, Z: torch.Tensor):
        noise = torch.randn((Z.shape[0], 1), device=Z.device)
        return self.gen(torch.cat([Z, noise], dim=1))


class Discriminator(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, ndfs: list[int]):
        super().__init__()
        self.disc = []
        last_dim = data_dim + z_dim
        for d in ndfs:
            self.disc.append(nn.Linear(last_dim, d))
            self.disc.append(nn.LeakyReLU(0.2))
            last_dim = d
        self.disc.append(nn.Linear(last_dim, 1))
        self.disc = nn.Sequential(*self.disc)
        self.apply(weights_init)

    def forward(self, X: torch.Tensor, Z: torch.Tensor):
        return self.disc(torch.cat([X, Z], dim=1))


class Reconstructor(nn.Module):
    def __init__(self, z_dim: int, data_dim: int, nrfs: list[int]):
        super().__init__()
        self.rec = []
        last_dim = data_dim
        for d in nrfs:
            self.rec.append(nn.Linear(last_dim, d))
            self.rec.append(nn.LeakyReLU(0.2))
            last_dim = d
        self.rec.append(nn.Linear(last_dim, z_dim))
        self.rec = nn.Sequential(*self.rec)
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        mus = self.rec(X)  # self.rec only generate mean values of gaussians
        return torch.randn(mus.shape, device=X.device, requires_grad=True) + mus


class VEEGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        self.R, self.optimizerR = None, None
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.data_dim, ngfs=self.config['ngfs'])
        self.D = Discriminator(self.config['z_dim'], self.data_dim, ndfs=self.config['ndfs'])
        self.R = Reconstructor(self.config['z_dim'], self.data_dim, nrfs=self.config['nrfs'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)
        self.R.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        self.optimizerR = optim.Adam(self.R.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])

    def define_losses(self):
        setattr(self, 'BCEWithLogits', nn.BCEWithLogitsLoss())
        setattr(self, 'MSE', nn.MSELoss())

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.R.load_state_dict(ckpt['R'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)
        self.R.to(device=self.device)

    def save_model(self, model_path):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict(), 'R': self.R.state_dict()}, model_path)

    def train_batch(self, ep, it, X, y=None):
        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)
        BCEWithLogits = getattr(self, 'BCEWithLogits')
        MSE = getattr(self, 'MSE')

        # --------- train discriminator --------- #
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fakeX = self.G(z).detach()
        rec_X = self.R(X).detach()
        d_realz_fakex = self.D(fakeX, z)
        d_fakez_realx = self.D(X, rec_X)
        lossD = (BCEWithLogits(d_realz_fakex, torch.ones_like(d_realz_fakex)) +
                 BCEWithLogits(d_fakez_realx, torch.zeros_like(d_fakez_realx))) / 2
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fakeX = self.G(z)
            rec_fakeX = self.R(fakeX)
            d_realz_fakex = self.D(fakeX, z)
            rec_error = MSE(rec_fakeX, z)
            lossGR = d_realz_fakex.mean() + rec_error.mean()
            self.optimizerG.zero_grad()
            self.optimizerR.zero_grad()
            lossGR.backward()
            self.optimizerG.step()
            self.optimizerR.step()
            self.writer.add_scalar('GR/loss', lossGR.item(), it + ep * len(self.dataloader))


def _test():
    G = Generator(z_dim=100, data_dim=2, ngfs=[256, 256])
    D = Discriminator(z_dim=100, data_dim=2, ndfs=[256, 256])
    R = Reconstructor(z_dim=100, data_dim=2, nrfs=[256, 256])
    z = torch.randn(10, 100)
    fakeX = G(z)
    score = D(z, fakeX)
    reconstructedz = R(fakeX)
    print(fakeX.shape)
    print(score.shape)
    print(reconstructedz.shape)


if __name__ == '__main__':
    _test()

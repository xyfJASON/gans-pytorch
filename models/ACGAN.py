import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_channels: int, n_classes: int):
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim+n_classes, 1024, (4, 4), stride=(1, 1), padding=(0, 0)),  # 4x4
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

    def forward(self, X: torch.Tensor, C: torch.Tensor):
        if X.ndim == 2:
            X = X.view(-1, X.shape[1], 1, 1)
        if C.ndim == 2:
            C = C.view(-1, C.shape[1], 1, 1)
        assert X.shape == (X.shape[0], self.z_dim, 1, 1)
        assert C.shape == (X.shape[0], self.n_classes, 1, 1)
        out = self.gen(torch.cat([X, C], dim=1))
        return out


class Discriminator(nn.Module):
    def __init__(self, img_channels: int, n_classes: int):
        super().__init__()
        self.ls = nn.Sequential(
            nn.Conv2d(img_channels, 128, (4, 4), stride=(2, 2), padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, (4, 4), stride=(2, 2), padding=1),  # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(1024, n_classes, (4, 4), stride=(1, 1), padding=0),  # 1x1
            nn.Flatten(),
        )
        self.disc = nn.Sequential(
            nn.Conv2d(1024, 1, (4, 4), stride=(1, 1), padding=0),  # 1x1
            nn.Flatten(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        X = self.ls(X)
        return self.disc(X), self.classifier(X)


class ACGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, conditional=True)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.img_channels, self.n_classes)
        self.D = Discriminator(self.img_channels, self.n_classes)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizer']['adam']['lr'], betas=self.config['optimizer']['adam']['betas'])

    def define_losses(self):
        setattr(self, 'BCEWithLogits', nn.BCEWithLogitsLoss())
        setattr(self, 'CrossEntropy', nn.CrossEntropyLoss())

    def train_batch(self, ep, it, X, y=None):
        assert X.shape[-2:] == (64, 64), f'ACGAN only supports 64x64 input.'

        X = X.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.long)
        y_onehot = F.one_hot(y, num_classes=self.n_classes).view(y.shape[0], -1).to(device=self.device)
        BCEWithLogits = getattr(self, 'BCEWithLogits')
        CrossEntropy = getattr(self, 'CrossEntropy')

        # --------- train discriminator --------- #
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z, y_onehot).detach()
        realscore, realclass = self.D(X)
        fakescore, fakeclass = self.D(fake)
        loss_score = (BCEWithLogits(realscore, torch.ones_like(realscore)) +
                      BCEWithLogits(fakescore, torch.zeros_like(fakescore)))
        loss_class = CrossEntropy(realclass, y) + CrossEntropy(fakeclass, y)
        lossD = loss_score + loss_class
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss_score', loss_score.item(), it + ep * len(self.dataloader))
        self.writer.add_scalar('D/loss_class', loss_class.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z, y_onehot)
            fakescore, fakeclass = self.D(fake)
            loss_score = BCEWithLogits(fakescore, torch.ones_like(fakescore))
            loss_class = CrossEntropy(fakeclass, y)
            lossG = loss_score + loss_class
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss_score', loss_score.item(), it + ep * len(self.dataloader))
            self.writer.add_scalar('G/loss_class', loss_class.item(), it + ep * len(self.dataloader))


def _test():
    G = Generator(z_dim=100, img_channels=3, n_classes=10)
    D = Discriminator(img_channels=3, n_classes=10)
    z = torch.randn(5, 100)
    c = torch.randint(0, 10, (5, ))
    c = F.one_hot(c, num_classes=10)
    fakeX = G(z, c)
    score = D(fakeX)
    print(fakeX.shape)
    print(score[0].shape, score[1].shape)
    print(sum([p.numel() for p in G.parameters()]))
    print(sum([p.numel() for p in D.parameters()]))


if __name__ == '__main__':
    _test()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from train import BaseTrainer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, k: int = 2):
        super().__init__()
        mid_channels = in_channels // k

        self.f = nn.Conv2d(in_channels, mid_channels, (1, 1))
        self.g = nn.Conv2d(in_channels, mid_channels, (1, 1))
        self.h = nn.Conv2d(in_channels, mid_channels, (1, 1))
        self.v = nn.Conv2d(mid_channels, in_channels, (1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, X: torch.Tensor):
        batchsize, C, H, W = X.shape
        fX = self.f(X).view(batchsize, -1, H*W)  # [batchsize, C/k, N]
        gX = self.g(X).view(batchsize, -1, H*W)  # [batchsize, C/k, N]
        hX = self.h(X).view(batchsize, -1, H*W)  # [batchsize, C/k, N]
        attention_map = F.softmax(torch.bmm(fX.permute(0, 2, 1), gX), dim=-1)  # [batchsize, N, N]
        output = torch.bmm(hX, attention_map.permute(0, 2, 1)).view(batchsize, -1, H, W)
        output = self.v(output)
        return self.gamma * output + X, attention_map


class Generator(nn.Module):
    def __init__(self, z_dim: int, img_channels: int, return_attmap: bool = True):
        super().__init__()
        self.return_attmap = return_attmap

        self.layer1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(z_dim, 512, (4, 4), stride=(1, 1), padding=(0, 0))),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1))),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1))),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.attn1 = SelfAttentionBlock(128)
        self.layer4 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1))),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.attn2 = SelfAttentionBlock(64)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        if X.ndim == 2:
            X = X.view(-1, X.shape[1], 1, 1)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X, attmap1 = self.attn1(X)
        X = self.layer4(X)
        X, attmap2 = self.attn2(X)
        X = self.layer5(X)
        if self.return_attmap:
            return X, attmap1, attmap2
        else:
            return X


class Discriminator(nn.Module):
    def __init__(self, img_channels: int):
        super().__init__()
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(img_channels, 64, (4, 4), stride=(2, 2), padding=1)),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn1 = SelfAttentionBlock(64)
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1)),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.attn2 = SelfAttentionBlock(128)
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1)),  # 8x8
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1)),  # 4x4
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 1, (4, 4), stride=(1, 1), padding=0)),  # 1x1
            nn.Flatten(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor):
        X = self.layer1(X)
        X, attmap1 = self.attn1(X)
        X = self.layer2(X)
        X, attmap2 = self.attn2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.layer5(X)
        return X


class SAGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        self.G = Generator(self.config['z_dim'], self.img_channels, return_attmap=False)
        self.D = Discriminator(self.img_channels)
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def define_optimizers(self):
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['optimizerG']['adam']['lr'], betas=self.config['optimizerG']['adam']['betas'])
        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.config['optimizerD']['adam']['lr'], betas=self.config['optimizerD']['adam']['betas'])

    def define_losses(self):
        pass

    def train_batch(self, ep, it, X, y=None):
        X = X.to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min E[max(0, 1 - D(X))] + E[max(0, 1 + D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = self.G(z).detach()
        d_real, d_fake = self.D(X), self.D(fake)
        lossD = torch.mean(F.relu(1 - d_real) + F.relu(1 + d_fake))
        self.optimizerD.zero_grad()
        lossD.backward()
        self.optimizerD.step()
        self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))
        self.writer.add_scalar('D/gamma1', self.D.attn1.gamma.item(), it + ep * len(self.dataloader))
        self.writer.add_scalar('D/gamma2', self.D.attn2.gamma.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # min -D(G(z))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = self.G(z)
            lossG = -torch.mean(self.D(fake))
            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()
            self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))
            self.writer.add_scalar('G/gamma1', self.G.attn1.gamma.item(), it + ep * len(self.dataloader))
            self.writer.add_scalar('G/gamma2', self.G.attn2.gamma.item(), it + ep * len(self.dataloader))


def _test():
    G = Generator(z_dim=100, img_channels=3)
    D = Discriminator(img_channels=3)
    z = torch.randn(10, 100, 1, 1)
    fakeX, attmap1, attmap2 = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(score.shape)
    print(attmap1.shape)
    print(attmap2.shape)
    print(sum([param.numel() for param in G.parameters()]))
    print(sum([param.numel() for param in D.parameters()]))


if __name__ == '__main__':
    _test()

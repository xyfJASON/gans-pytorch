import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, n_classes: int, img_channels: int) -> None:
        super().__init__()
        self.ls = nn.Sequential(
            nn.Conv2d(img_channels, 128, (4, 4), stride=(2, 2), padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1),  # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1),  # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(512, 1024, (4, 4), stride=(2, 2), padding=1),  # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
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

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.ls(X)
        return self.disc(X), self.classifier(X)


class Generator(nn.Module):
    def __init__(self, z_dim: int, n_classes: int, img_channels: int) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.n_classes = n_classes
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim+n_classes, 1024, (4, 4), stride=(1, 1), padding=(0, 0)),  # 4x4
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, (4, 4), stride=(2, 2), padding=(1, 1)),  # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1)),  # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),  # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        batchsize = X.shape[0]
        assert X.shape == (batchsize, self.z_dim, 1, 1)
        assert C.shape == (batchsize, self.n_classes, 1, 1)
        out = self.gen(torch.cat([X, C], dim=1))
        return out


if __name__ == '__main__':
    import torch.nn.functional as F
    G = Generator(z_dim=100, n_classes=10, img_channels=3)
    D = Discriminator(n_classes=10, img_channels=3)
    z = torch.randn(5, 100, 1, 1)
    c = torch.randint(0, 10, (5, ))
    c = F.one_hot(c, num_classes=10).view(5, -1, 1, 1)
    fakeX = G(z, c)
    score = D(fakeX)
    print(fakeX.shape)
    print(score[0].shape, score[1].shape)
    print(sum([p.numel() for p in G.parameters()]))
    print(sum([p.numel() for p in D.parameters()]))

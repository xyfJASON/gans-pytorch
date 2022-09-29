import torch
import torch.nn as nn
import torch.nn.functional as F


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

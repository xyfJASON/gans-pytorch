import torch
from torch.utils.data import Dataset


class Scaler:
    def __init__(self, datamin, datamax):
        self.datamin, self.datamax = datamin, datamax

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.datamin) / (self.datamax - self.datamin + 1e-12) * 2 - 1

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X + 1) / 2 * (self.datamax - self.datamin + 1e-12) + self.datamin


class Ring8(Dataset):
    def __init__(self):
        self.n_each = 400
        self.x = torch.randn((self.n_each, 8)) * 0.5
        self.y = torch.randn((self.n_each, 8)) * 0.5
        angles = torch.linspace(0, 1.75, 8) * 3.1415926
        self.x = (self.x + 10 * torch.cos(angles)).view(-1)
        self.y = (self.y + 10 * torch.sin(angles)).view(-1)
        self.scaler = Scaler(torch.tensor([-15, -15]), torch.tensor([15, 15]))

    def __len__(self):
        return 8 * self.n_each

    def __getitem__(self, item):
        res = torch.tensor([self.x[item], self.y[item]])
        res = self.scaler.transform(res)
        return res


class Grid25(Dataset):
    def __init__(self):
        self.n_each = 400
        self.x = torch.randn((self.n_each, 5, 5)) * 0.5
        self.y = torch.randn((self.n_each, 5, 5)) * 0.5
        biasx, biasy = torch.meshgrid(torch.linspace(-10, 10, 5), torch.linspace(-10, 10, 5), indexing='ij')
        self.x = (self.x + biasx).view(-1)
        self.y = (self.y + biasy).view(-1)
        self.scaler = Scaler(torch.tensor([-15, -15]), torch.tensor([15, 15]))

    def __len__(self):
        return 25 * self.n_each

    def __getitem__(self, item):
        res = torch.tensor([self.x[item], self.y[item]])
        res = self.scaler.transform(res)
        return res


if __name__ == '__main__':
    grid = Grid25()
    data = torch.stack([grid[i] for i in range(len(grid))])
    data = grid.scaler.inverse_transform(data)
    import matplotlib.pyplot as plt
    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.axis('equal')
    plt.show()
    plt.close()

    ring = Ring8()
    data = torch.stack([ring[i] for i in range(len(ring))])
    data = grid.scaler.inverse_transform(data)
    plt.scatter(data[:, 0], data[:, 1], s=1)
    plt.axis('equal')
    plt.show()
    plt.close()

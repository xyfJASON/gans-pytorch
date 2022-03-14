import torch
from torchvision.utils import save_image

import models


class Tester:
    def __init__(self, model_path: str, z_dim: int, data_dim: int, use_gpu: bool = True):
        ckpt = torch.load(model_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print('using device:', self.device)
        self.z_dim = z_dim
        self.data_dim = data_dim

        self.G = models.Generator(z_dim, data_dim)
        self.G.load_state_dict(ckpt['G'])
        self.G.to(device=self.device)

    @torch.no_grad()
    def walk_in_latent_space(self, save_path):
        self.G.eval()
        z1 = torch.randn((8, self.z_dim), device=self.device)
        z2 = torch.randn((8, self.z_dim), device=self.device)
        result = []
        for t in torch.linspace(0, 1, 15):
            imgs = self.G(z1*t+z2*(1-t)).cpu()
            result.append(imgs)
        result = torch.stack(result, dim=1).reshape(8*15, 1, 28, 28)
        save_image(result, save_path, nrow=15, normalize=True, value_range=(-1, 1))

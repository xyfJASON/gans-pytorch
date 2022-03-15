import torch
from torchvision.utils import save_image

import models


class Tester:
    def __init__(self, model_path: str, z_dim: int, c_dim: int, img_channels: int, use_gpu: bool = True):
        ckpt = torch.load(model_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print('using device:', self.device)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.img_channels = img_channels

        self.G = models.Generator(z_dim, c_dim, img_channels)
        self.G.load_state_dict(ckpt['G'])
        self.G.to(device=self.device)

    @torch.no_grad()
    def fix_z(self, save_path):
        self.G.eval()
        z = torch.randn((8, 1, self.z_dim), device=self.device)
        z = z.expand((8, self.c_dim, self.z_dim)).reshape(8*self.c_dim, self.z_dim, 1, 1)
        c = torch.arange(0, self.c_dim, device=self.device).repeat(8)
        imgs = self.G(z, c).cpu().reshape(8*self.c_dim, self.img_channels, 64, 64)
        save_image(imgs, save_path, nrow=self.c_dim, normalize=True, value_range=(-1, 1))

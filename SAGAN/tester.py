from PIL import ImageDraw
import torch
import torchvision.transforms as T
from torchvision.utils import save_image

import models


class Tester:
    def __init__(self, model_path: str, z_dim: int, img_channels: int, use_gpu: bool = True):
        ckpt = torch.load(model_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print('using device:', self.device)
        self.z_dim = z_dim
        self.img_channels = img_channels

        self.G = models.Generator(z_dim, img_channels)
        self.G.load_state_dict(ckpt['G'])
        self.G.to(device=self.device)

    @torch.no_grad()
    def walk_in_latent_space(self, save_path):
        self.G.eval()
        z1 = torch.randn((8, self.z_dim, 1, 1), device=self.device)
        z2 = torch.randn((8, self.z_dim, 1, 1), device=self.device)
        result = []
        for t in torch.linspace(0, 1, 15):
            imgs = self.G(z1*t+z2*(1-t))[0].cpu()
            result.append(imgs)
        result = torch.stack(result, dim=1).reshape(8*15, self.img_channels, 64, 64)
        save_image(result, save_path, nrow=15, normalize=True, value_range=(-1, 1))

    @torch.no_grad()
    def random_sample_with_attmaps(self, n_imgs: int, save_path):
        self.G.eval()
        z = torch.randn((n_imgs, self.z_dim, 1, 1), device=self.device)
        imgs, _, attmaps = self.G(z)
        result = []
        colors = ['red', 'cyan', 'green', 'magenta']
        for img, attmap in zip(imgs, attmaps):
            pos = torch.randint(8, 56, (4, 2))
            result.append(self._draw_dot((img + 1) / 2, pos, colors))
            for i in range(4):
                spatial_attmap = self._get_pixel_attmap(attmap, pos[i])
                result.append(self._draw_dot(spatial_attmap, pos[i:i+1], colors[i:i+1]))
        save_image(result, save_path, nrow=5, normalize=True, value_range=(0, 1))

    @staticmethod
    def _get_pixel_attmap(attmap: torch.Tensor, pos):
        if isinstance(pos, torch.Tensor):
            pos = pos.tolist()
        assert attmap.shape == (1024, 1024)
        assert pos[0] < 64 and pos[1] < 64
        pos = pos[0] // 2 * 32 + pos[1] // 2
        pixel_attmap = attmap[pos].view(32, 32).unsqueeze(0)
        pixel_attmap = T.Resize((64, 64))(pixel_attmap)
        pixel_attmap /= pixel_attmap.sum()
        return pixel_attmap  # [1, 64, 64]

    @staticmethod
    def _draw_dot(img: torch.Tensor, pos, colors):
        assert (0 <= img).all() and (img <= 1).all()
        if isinstance(pos, torch.Tensor):
            pos = pos.tolist()
        img = img.expand(3, 64, 64)
        image = T.ToPILImage()(img)
        draw = ImageDraw.Draw(image)
        for i in range(len(pos)):
            draw.ellipse([pos[i][0]-1, pos[i][1]-1, pos[i][0]+1, pos[i][1]+1], fill=colors[i], outline=None, width=0)
        return T.ToTensor()(image)

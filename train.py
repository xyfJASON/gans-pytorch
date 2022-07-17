import os
import yaml
import shutil
import imageio
import datetime
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.utils import save_image

import models
from dataset import Ring8, Grid25
from utils.general_utils import makedirs


class BaseTrainer:
    def __init__(self, config_path: str, conditional: bool = False):
        # ====================================================== #
        # CONFIGURATIONS
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.log_root = os.path.join('runs', datetime.datetime.now().strftime('exp-%Y-%m-%d-%H-%M-%S'))
        print('log directory:', self.log_root)

        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        print('using device:', self.device)

        if self.config.get('save_per_epochs'):
            makedirs(os.path.join(self.log_root, 'ckpt'))

        makedirs(os.path.join(self.log_root, 'tensorboard'))
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        if self.config.get('sample_per_epochs'):
            makedirs(os.path.join(self.log_root, 'samples'))

        if not os.path.exists(os.path.join(self.log_root, 'config.yml')):
            shutil.copyfile(config_path, os.path.join(self.log_root, 'config.yml'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        self.conditional = conditional

        print('==> Getting data...')
        if self.config['dataset'] == 'ring8':
            self.dataset = Ring8()
            self.data_dim = 2
            self.img_channels = None
            self.n_classes = None
        elif self.config['dataset'] == 'grid25':
            self.dataset = Grid25()
            self.data_dim = 2
            self.img_channels = None
            self.n_classes = None
        elif self.config['dataset'] == 'mnist':
            transforms = T.Compose([T.Resize((self.config['img_size'], self.config['img_size'])), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            self.dataset = dset.MNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            self.data_dim = self.config['img_size'] * self.config['img_size']
            self.img_channels = 1
            self.n_classes = 10
        elif self.config['dataset'] == 'fashion_mnist':
            transforms = T.Compose([T.Resize((self.config['img_size'], self.config['img_size'])), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
            self.dataset = dset.FashionMNIST(root=self.config['dataroot'], train=True, transform=transforms, download=False)
            self.data_dim = self.config['img_size'] * self.config['img_size']
            self.img_channels = 1
            self.n_classes = 10
        elif self.config['dataset'] == 'celeba':
            transforms = T.Compose([T.Resize((self.config['img_size'], self.config['img_size'])), T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
            self.dataset = dset.CelebA(root=self.config['dataroot'], split='train', transform=transforms, download=False)
            self.data_dim = self.config['img_size'] * self.config['img_size']
            self.img_channels = 3
            self.n_classes = None
        else:
            raise ValueError(f"Dataset {self.config['dataset']} is not supported now.")

        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        # ====================================================== #
        # DEFINE MODELS, OPTIMIZERS, etc.
        # ====================================================== #
        self.G, self.D = None, None
        self.optimizerG, self.optimizerD = None, None
        self.define_models()
        self.define_optimizers()
        self.define_losses()

        # ====================================================== #
        # TEST SAMPLES
        # ====================================================== #
        if not conditional:
            if self.config['dataset'] in ['ring8', 'grid25']:
                self.sample_z = torch.randn((1000, self.config['z_dim']), device=self.device)
            else:
                self.sample_z = torch.randn((64, self.config['z_dim']), device=self.device)
        else:
            if not self.n_classes:
                raise ValueError(f"Dataset {self.config['dataset']} does not support conditional generation.")
            self.sample_z = torch.randn((5 * self.n_classes, self.config['z_dim']), device=self.device)
            self.sample_c = F.one_hot(torch.arange(0, self.n_classes, device=self.device).repeat(5, ), num_classes=self.n_classes)

    def define_models(self):
        raise NotImplementedError

    def define_optimizers(self):
        raise NotImplementedError

    def define_losses(self):
        raise NotImplementedError

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def save_model(self, model_path):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict()}, model_path)

    def train(self):
        print('==> Training...')
        sample_paths = []
        for ep in range(self.config['epochs']):
            self.G.train()
            self.D.train()

            for it, X in enumerate(tqdm(self.dataloader, desc=f'Epoch {ep}', ncols=120)):
                if isinstance(X, tuple) or isinstance(X, list):
                    self.train_batch(ep, it, X[0], X[1])
                else:
                    self.train_batch(ep, it, X)

            if self.config.get('sample_per_epochs') and (ep + 1) % self.config['sample_per_epochs'] == 0:
                self.sample_generator(ep, os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))
                sample_paths.append(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.generate_gif(sample_paths, os.path.join(self.log_root, f'samples.gif'))
        self.writer.close()

    def train_batch(self, ep, it, X, y=None):
        raise NotImplementedError

    @torch.no_grad()
    def sample_generator(self, ep, savepath):
        self.G.eval()
        if self.config['dataset'] in ['ring8', 'grid25']:
            X = self.G(self.sample_z).cpu()
            realX = torch.stack([d for d in self.dataset], dim=0)  # noqa
            realX = self.dataset.scaler.inverse_transform(realX)
            X = self.dataset.scaler.inverse_transform(X)
            fig, ax = plt.subplots(1, 1)
            ax.scatter(realX[:, 0], realX[:, 1], c='green', s=1, alpha=0.5)
            ax.scatter(X[:, 0], X[:, 1], c='blue', s=1)
            ax.axis('scaled')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.set_title(f'Epoch {ep}')
            fig.savefig(savepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
        elif self.config['dataset'] in ['mnist', 'fashion_mnist', 'celeba']:
            if not self.conditional:
                X = self.G(self.sample_z).cpu()
                X = X.view(-1, self.img_channels, self.config['img_size'], self.config['img_size'])
                save_image(X, savepath, nrow=8, normalize=True, value_range=(-1, 1))
            else:
                X = self.G(self.sample_z, self.sample_c).cpu()
                X = X.view(-1, self.img_channels, self.config['img_size'], self.config['img_size'])
                save_image(X, savepath, nrow=self.n_classes, normalize=True, value_range=(-1, 1))
        else:
            raise ValueError

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    parser.add_argument('--model', choices=['gan',
                                            'dcgan',
                                            'cgan',
                                            'acgan',
                                            'wgan',
                                            'wgan-gp',
                                            'sngan',
                                            'sngan-projection',
                                            'lsgan',
                                            'sagan',
                                            'veegan',
                                            ], required=True, help='choose a gan to train')
    args = parser.parse_args()

    if args.model == 'gan':
        trainer = models.GAN.GAN_Trainer(args.config_path)
    elif args.model == 'dcgan':
        trainer = models.DCGAN.DCGAN_Trainer(args.config_path)
    elif args.model == 'cgan':
        trainer = models.CGAN.CGAN_Trainer(args.config_path)
    elif args.model == 'acgan':
        trainer = models.ACGAN.ACGAN_Trainer(args.config_path)
    elif args.model == 'wgan':
        trainer = models.WGAN.WGAN_Trainer(args.config_path)
    elif args.model == 'wgan-gp':
        trainer = models.WGAN_GP.WGAN_GP_Trainer(args.config_path)
    elif args.model == 'sngan':
        trainer = models.SNGAN.SNGAN_Trainer(args.config_path)
    elif args.model == 'sngan-projection':
        trainer = models.SNGAN_projection.SNGAN_projection_Trainer(args.config_path)
    elif args.model == 'lsgan':
        trainer = models.LSGAN.LSGAN_Trainer(args.config_path)
    elif args.model == 'sagan':
        trainer = models.SAGAN.SAGAN_Trainer(args.config_path)
    elif args.model == 'veegan':
        trainer = models.VEEGAN.VEEGAN_Trainer(args.config_path)
    else:
        raise ValueError(f'{args.model} is not supported now.')

    trainer.train()

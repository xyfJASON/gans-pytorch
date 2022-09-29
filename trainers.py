import os
import yaml
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.autograd as autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import models
from utils.optimizer import build_optimizer
from utils.data import build_dataset, build_dataloader
from utils.train_utils import set_device, create_log_directory, reduce_tensor


class BaseTrainer:
    def __init__(self, config_path: str, conditional: bool = False):
        # ====================================================== #
        # READ CONFIGURATION FILE
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ====================================================== #
        # SET DEVICE
        # ====================================================== #
        self.device, self.world_size, self.local_rank, self.global_rank = set_device()
        self.is_master = self.world_size <= 1 or self.global_rank == 0
        self.is_ddp = self.world_size > 1
        print('using device:', self.device)

        # ====================================================== #
        # CREATE LOG DIRECTORY
        # ====================================================== #
        if self.is_master:
            self.log_root = create_log_directory(self.config, config_path)

        # ====================================================== #
        # TENSORBOARD
        # ====================================================== #
        if self.is_master:
            os.makedirs(os.path.join(self.log_root, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        self.dataset, self.data_dim, self.img_channels, self.n_classes = build_dataset(self.config['dataset'], dataroot=self.config['dataroot'], img_size=self.config['img_size'], split='train')
        self.dataloader = build_dataloader(self.dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, is_ddp=self.is_ddp)

        # ====================================================== #
        # DEFINE MODELS, OPTIMIZERS, etc.
        # ====================================================== #
        self.define_models()
        self.define_optimizers()
        self.define_losses()

        # ====================================================== #
        # TEST SAMPLES
        # ====================================================== #
        self.conditional = conditional
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
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG = build_optimizer(G.parameters(), cfg=self.config['optimizer'])
        optimizerD = build_optimizer(D.parameters(), cfg=self.config['optimizer'])
        setattr(self, 'optimizerG', optimizerG)
        setattr(self, 'optimizerD', optimizerD)

    def define_losses(self):
        raise NotImplementedError

    def load_model(self, model_path):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        ckpt = torch.load(model_path, map_location='cpu')
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        G.to(device=self.device)
        D.to(device=self.device)

    def save_model(self, model_path):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        G = G.module if self.is_ddp else G
        D = D.module if self.is_ddp else D
        torch.save({'G': G.state_dict(), 'D': D.state_dict()}, model_path)

    def train(self):
        print('==> Training...')
        sample_paths = []
        for ep in range(self.config['epochs']):
            if self.is_ddp:
                dist.barrier()
                self.dataloader.sampler.set_epoch(ep)

            self.train_one_epoch(ep)

            if self.is_master:
                if self.config.get('sample_freq') and (ep + 1) % self.config['sample_freq'] == 0:
                    self.sample(ep, os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))
                    sample_paths.append(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

        if self.is_master:
            self.save_model(os.path.join(self.log_root, 'model.pt'))
            self.generate_gif(sample_paths, os.path.join(self.log_root, f'samples.gif'))
            self.writer.close()

    def train_one_epoch(self, ep):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        G.train(); D.train()
        pbar = tqdm(self.dataloader, desc=f'Epoch {ep}', ncols=120) if self.is_master else self.dataloader
        for it, X in enumerate(pbar):
            self.train_batch(ep, it, X[0], X[1]) if isinstance(X, (tuple, list)) else self.train_batch(ep, it, X)
        if self.is_master:
            pbar.close()

    def train_batch(self, ep, it, X, y=None):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, ep, savepath):
        G = getattr(self, 'G').module if self.is_ddp else getattr(self, 'G')
        G.eval()
        if self.config['dataset'] in ['ring8', 'grid25']:
            X = G(self.sample_z).cpu()
            realX = torch.stack([d for d in self.dataset], dim=0)
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
                X = G(self.sample_z).cpu()
                X = X.view(-1, self.img_channels, self.config['img_size'], self.config['img_size'])
                save_image(X, savepath, nrow=8, normalize=True, value_range=(-1, 1))
            else:
                X = G(self.sample_z, self.sample_c).cpu()
                X = X.view(-1, self.img_channels, self.config['img_size'], self.config['img_size'])
                save_image(X, savepath, nrow=self.n_classes, normalize=True, value_range=(-1, 1))
        else:
            raise ValueError

    @staticmethod
    def generate_gif(img_paths, savepath, duration=0.1):
        images = [imageio.imread(p) for p in img_paths]
        imageio.mimsave(savepath, images, 'GIF', duration=duration)


# ============================================ GAN ============================================ #
class GAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.GAN.Generator(self.config['z_dim'], self.data_dim)
        D = models.GAN.Discriminator(self.data_dim)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        BCE = getattr(self, 'BCE')

        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min -[log(D(x)) + log(1-D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        realscore, fakescore = D(X), D(fake)
        lossD = (BCE(realscore, torch.ones_like(realscore)) + BCE(fakescore, torch.zeros_like(fakescore))) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z))) => min -log(D(G(z)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            fakescore = D(fake)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ DCGAN ============================================ #
class DCGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.DCGAN.Generator(self.config['z_dim'], self.img_channels)
        D = models.DCGAN.Discriminator(self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        BCE = getattr(self, 'BCE')

        assert X.shape[-2:] == (64, 64), f'DCGAN only supports 64x64 input.'
        X = X.to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min -[log(D(x)) + log(1-D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        realscore, fakescore = D(X), D(fake)
        lossD = (BCE(realscore, torch.ones_like(realscore)) + BCE(fakescore, torch.zeros_like(fakescore))) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z))) => min -log(D(G(z)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            fakescore = D(fake)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ WGAN ============================================ #
class WGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.WGAN.Generator(self.config['z_dim'], self.data_dim)
        D = models.WGAN.Discriminator(self.data_dim)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        pass

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')

        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # max E[D(x)]-E[D(G(z))] <=> min E[D(G(z))]-E[D(x)]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        lossD = torch.mean(D(fake)) - torch.mean(D(X))
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        for param in D.parameters():
            param.data.clamp_(min=self.config['clip'][0], max=self.config['clip'][1])  # weight clipping

        # --------- train generator --------- #
        # max E[D(G(z))] <=> min E[-D(G(z))]
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            lossG = -torch.mean(D(fake))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ WGAN-GP ============================================ #
class WGAN_GP_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.WGAN_GP.Generator(self.config['z_dim'], self.data_dim)
        D = models.WGAN_GP.Discriminator(self.data_dim)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        pass

    def gradient_penalty(self, realX: torch.Tensor, fakeX: torch.Tensor):
        D = getattr(self, 'D')
        alpha = torch.rand(1, device=self.device)
        interX = alpha * realX + (1 - alpha) * fakeX
        interX.requires_grad_()
        d_interX = D(interX)
        gradients = autograd.grad(outputs=d_interX, inputs=interX,
                                  grad_outputs=torch.ones_like(d_interX),
                                  create_graph=True, retain_graph=True)[0]
        gradients = gradients.flatten(start_dim=1)
        return torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')

        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min E[D(G(z))] - E[D(x)] + lambda * gp
        D.zero_grad()
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        lossD = torch.mean(D(fake)) - torch.mean(D(X)) + self.config['lambda_gp'] * self.gradient_penalty(X, fake)
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max E[D(G(z))] <=> min E[-D(G(z))]
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            lossG = -torch.mean(D(fake))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ LSGAN ============================================ #
class LSGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        if self.config['model_arch'] == 'MLP':
            G = models.LSGAN.GeneratorMLP(self.config['z_dim'], self.data_dim)
            D = models.LSGAN.DiscriminatorMLP(self.data_dim)
        elif self.config['model_arch'] == 'CNN':
            G = models.LSGAN.GeneratorCNN(self.config['z_dim'], self.img_channels)
            D = models.LSGAN.DiscriminatorCNN(self.img_channels)
        else:
            raise ValueError('model architecture should be either MLP or CNN.')
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'MSE', nn.MSELoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        MSE = getattr(self, 'MSE')

        X = X.flatten(start_dim=1) if self.config['model_arch'] == 'MLP' else X
        X = X.to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min [(D(x)-b)^2 + (D(G(z))-a)^2] / 2
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        realscore, fakescore = D(X), D(fake)
        lossD = (MSE(realscore, torch.ones_like(realscore) * self.config['b']) +
                 MSE(fakescore, torch.ones_like(fakescore) * self.config['a'])) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # min (D(G(z))-c)^2 / 2
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            fakescore = D(fake)
            lossG = MSE(fakescore, torch.ones_like(fakescore) * self.config['c']) / 2
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ SNGAN ============================================ #
class SNGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.SNGAN.Generator(self.config['z_dim'], self.img_channels)
        D = models.SNGAN.Discriminator(self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        BCE = getattr(self, 'BCE')

        assert X.shape[-2:] == (64, 64), f'SNGAN only supports 64x64 input.'
        X = X.to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min -[log(D(x)) + log(1-D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        realscore, fakescore = D(X), D(fake)
        lossD = (BCE(realscore, torch.ones_like(realscore)) + BCE(fakescore, torch.zeros_like(fakescore))) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z))) => min -log(D(G(z)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            fakescore = D(fake)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ SAGAN ============================================ #
class SAGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.SAGAN.Generator(self.config['z_dim'], self.img_channels, return_attmap=False)
        D = models.SAGAN.Discriminator(self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_optimizers(self):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG = build_optimizer(G.parameters(), cfg=self.config['optimizerG'])
        optimizerD = build_optimizer(D.parameters(), cfg=self.config['optimizerD'])
        setattr(self, 'optimizerG', optimizerG)
        setattr(self, 'optimizerD', optimizerD)

    def define_losses(self):
        pass

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')

        X = X.to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        # min E[max(0, 1 - D(X))] + E[max(0, 1 + D(G(z)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z).detach()
        d_real, d_fake = D(X), D(fake)
        lossD = torch.mean(F.relu(1 - d_real) + F.relu(1 + d_fake))
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # min -D(G(z))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z)
            lossG = -torch.mean(D(fake))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ VEEGAN ============================================ #
class VEEGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)

    def define_models(self):
        G = models.VEEGAN.Generator(self.config['z_dim'], self.data_dim, ngfs=self.config['ngfs'])
        D = models.VEEGAN.Discriminator(self.config['z_dim'], self.data_dim, ndfs=self.config['ndfs'])
        R = models.VEEGAN.Reconstructor(self.config['z_dim'], self.data_dim, nrfs=self.config['nrfs'])
        G.to(device=self.device)
        D.to(device=self.device)
        R.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
            R = torch.nn.SyncBatchNorm.convert_sync_batchnorm(R)
            R = DDP(R, device_ids=[self.local_rank], output_device=self.local_rank)
        setattr(self, 'G', G)
        setattr(self, 'D', D)
        setattr(self, 'R', R)

    def define_optimizers(self):
        G, D, R = getattr(self, 'G'), getattr(self, 'D'), getattr(self, 'R')
        optimizerG = build_optimizer(G.parameters(), cfg=self.config['optimizer'])
        optimizerD = build_optimizer(D.parameters(), cfg=self.config['optimizer'])
        optimizerR = build_optimizer(R.parameters(), cfg=self.config['optimizer'])
        setattr(self, 'optimizerG', optimizerG)
        setattr(self, 'optimizerD', optimizerD)
        setattr(self, 'optimizerR', optimizerR)

    def define_losses(self):
        setattr(self, 'BCEWithLogits', nn.BCEWithLogitsLoss())
        setattr(self, 'MSE', nn.MSELoss())

    def load_model(self, model_path):
        G, D, R = getattr(self, 'G'), getattr(self, 'D'), getattr(self, 'R')
        ckpt = torch.load(model_path, map_location='cpu')
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        R.load_state_dict(ckpt['R'])
        G.to(device=self.device)
        D.to(device=self.device)
        R.to(device=self.device)

    def save_model(self, model_path):
        G, D, R = getattr(self, 'G'), getattr(self, 'D'), getattr(self, 'R')
        G = G.module if self.is_ddp else G
        D = D.module if self.is_ddp else D
        R = R.module if self.is_ddp else R
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'R': R.state_dict()}, model_path)

    def train_batch(self, ep, it, X, y=None):
        G, D, R = getattr(self, 'G'), getattr(self, 'D'), getattr(self, 'R')
        optimizerG, optimizerD, optimizerR = getattr(self, 'optimizerG'), getattr(self, 'optimizerD'), getattr(self, 'optimizerR')
        BCEWithLogits = getattr(self, 'BCEWithLogits')
        MSE = getattr(self, 'MSE')

        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)

        # --------- train discriminator --------- #
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fakeX = G(z).detach()
        rec_X = R(X).detach()
        d_realz_fakex = D(fakeX, z)
        d_fakez_realx = D(X, rec_X)
        lossD = (BCEWithLogits(d_realz_fakex, torch.ones_like(d_realz_fakex)) + BCEWithLogits(d_fakez_realx, torch.zeros_like(d_fakez_realx))) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fakeX = G(z)
            rec_fakeX = R(fakeX)
            d_realz_fakex = D(fakeX, z)
            rec_error = MSE(rec_fakeX, z)
            lossGR = d_realz_fakex.mean() + rec_error.mean()
            optimizerG.zero_grad()
            optimizerR.zero_grad()
            lossGR.backward()
            optimizerG.step()
            optimizerR.step()
            if self.is_ddp:
                lossGR = reduce_tensor(lossGR.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('GR/loss', lossGR.item(), it + ep * len(self.dataloader))


# ============================================ CGAN ============================================ #
class CGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, conditional=True)

    def define_models(self):
        G = models.CGAN.Generator(self.config['z_dim'], self.data_dim, self.n_classes)
        D = models.CGAN.Discriminator(self.data_dim, self.n_classes)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'BCE', nn.BCELoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        BCE = getattr(self, 'BCE')

        X = X.flatten(start_dim=1).to(device=self.device, dtype=torch.float32)
        y = F.one_hot(y, num_classes=self.n_classes).to(device=self.device)

        # --------- train discriminator --------- #
        # min -[log(D(x|y)) + log(1-D(G(z|y)))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z, y).detach()
        realscore, fakescore = D(X, y), D(fake, y)
        lossD = (BCE(realscore, torch.ones_like(realscore)) + BCE(fakescore, torch.zeros_like(fakescore))) / 2
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # max -log(1-D(G(z|y))) => min -log(D(G(z|y)))
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z, y)
            fakescore = D(fake, y)
            lossG = BCE(fakescore, torch.ones_like(fakescore))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))


# ============================================ ACGAN ============================================ #
class ACGAN_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, conditional=True)

    def define_models(self):
        G = models.ACGAN.Generator(self.config['z_dim'], self.img_channels, self.n_classes)
        D = models.ACGAN.Discriminator(self.img_channels, self.n_classes)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        setattr(self, 'BCEWithLogits', nn.BCEWithLogitsLoss())
        setattr(self, 'CrossEntropy', nn.CrossEntropyLoss())

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')
        BCEWithLogits = getattr(self, 'BCEWithLogits')
        CrossEntropy = getattr(self, 'CrossEntropy')

        assert X.shape[-2:] == (64, 64), f'ACGAN only supports 64x64 input.'
        X = X.to(device=self.device, dtype=torch.float32)
        y = y.to(device=self.device, dtype=torch.long)
        y_onehot = F.one_hot(y, num_classes=self.n_classes).view(y.shape[0], -1).to(device=self.device)

        # --------- train discriminator --------- #
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z, y_onehot).detach()
        realscore, realclass = D(X)
        fakescore, fakeclass = D(fake)
        loss_score = BCEWithLogits(realscore, torch.ones_like(realscore)) + BCEWithLogits(fakescore, torch.zeros_like(fakescore))
        loss_class = CrossEntropy(realclass, y) + CrossEntropy(fakeclass, y)
        lossD = loss_score + loss_class
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            loss_score = reduce_tensor(loss_score.detach(), self.world_size)
            loss_class = reduce_tensor(loss_class.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss_score', loss_score.item(), it + ep * len(self.dataloader))
            self.writer.add_scalar('D/loss_class', loss_class.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
            fake = G(z, y_onehot)
            fakescore, fakeclass = D(fake)
            loss_score = BCEWithLogits(fakescore, torch.ones_like(fakescore))
            loss_class = CrossEntropy(fakeclass, y)
            lossG = loss_score + loss_class
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                loss_score = reduce_tensor(loss_score.detach(), self.world_size)
                loss_class = reduce_tensor(loss_class.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss_score', loss_score.item(), it + ep * len(self.dataloader))
                self.writer.add_scalar('G/loss_class', loss_class.item(), it + ep * len(self.dataloader))


# ============================================ SNGAN-projection ============================================ #
class SNGAN_projection_Trainer(BaseTrainer):
    def __init__(self, config_path):
        super().__init__(config_path, conditional=True)

    def define_models(self):
        G = models.SNGAN_projection.Generator(self.config['z_dim'], self.n_classes, self.img_channels)
        D = models.SNGAN_projection.Discriminator(self.n_classes, self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        if self.is_ddp:
            G = torch.nn.SyncBatchNorm.convert_sync_batchnorm(G)
            G = DDP(G, device_ids=[self.local_rank], output_device=self.local_rank)
            D = torch.nn.SyncBatchNorm.convert_sync_batchnorm(D)
            D = DDP(D, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)
        setattr(self, 'G', G)
        setattr(self, 'D', D)

    def define_losses(self):
        pass

    def train_batch(self, ep, it, X, y=None):
        G, D = getattr(self, 'G'), getattr(self, 'D')
        optimizerG, optimizerD = getattr(self, 'optimizerG'), getattr(self, 'optimizerD')

        assert X.shape[-2:] == (64, 64), f'SNGAN-projection only supports 64x64 input.'
        X = X.to(device=self.device, dtype=torch.float32)
        y = F.one_hot(y, num_classes=self.n_classes).to(device=self.device)

        # --------- train discriminator --------- #
        # min E[max(0, 1 - D(X, y))] + E[max(0, 1 + D(G(z, y), y))]
        z = torch.randn((X.shape[0], self.config['z_dim']), device=self.device)
        fake = G(z, y).detach()
        d_real, d_fake = D(X, y), D(fake, y)
        lossD = torch.mean(F.relu(1 - d_real) + F.relu(1 + d_fake))
        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()
        if self.is_ddp:
            lossD = reduce_tensor(lossD.detach(), self.world_size)
        if self.is_master:
            self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.dataloader))

        # --------- train generator --------- #
        # min -D(G(z, y), y)
        if (it + 1) % self.config['d_iters'] == 0:
            z = torch.randn((X.shape[0], self.config['z_dim'], 1, 1), device=self.device)
            fake = G(z, y)
            lossG = -torch.mean(D(fake, y))
            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()
            if self.is_ddp:
                lossG = reduce_tensor(lossG.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('G/loss', lossG.item(), it + ep * len(self.dataloader))

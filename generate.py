import argparse
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

import models


@torch.no_grad()
def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.model_path, map_location='cpu')
    if args.model == 'gan':
        G = models.GAN.Generator(args.z_dim, args.data_dim)
    elif args.model == 'dcgan':
        G = models.DCGAN.Generator(args.z_dim, args.img_channels)
    elif args.model == 'cgan':
        G = models.CGAN.Generator(args.z_dim, args.data_dim, args.n_classes)
    elif args.model == 'acgan':
        G = models.ACGAN.Generator(args.z_dim, args.img_channels, args.n_classes)
    elif args.model == 'wgan':
        G = models.WGAN.Generator(args.z_dim, args.data_dim)
    elif args.model == 'wgan-gp':
        G = models.WGAN_GP.Generator(args.z_dim, args.data_dim)
    elif args.model == 'sngan':
        G = models.SNGAN.Generator(args.z_dim, args.img_channels)
    elif args.model == 'sngan-projection':
        G = models.SNGAN_projection.Generator(args.z_dim, args.n_classes, args.img_channels)
    elif args.model == 'lsgan':
        G = models.LSGAN.GeneratorCNN(args.z_dim, args.img_channels)
    elif args.model == 'sagan':
        G = models.SAGAN.Generator(args.z_dim, args.img_channels)
    elif args.model == 'veegan':
        G = models.VEEGAN.Generator(args.z_dim, args.data_dim, ngfs=[256, 256, 256])
    else:
        raise ValueError(f'{args.model} is not supported now.')
    G.load_state_dict(ckpt['G'])
    G.to(device=device)

    if args.mode == 'random':
        if not args.conditional:
            sample_z = torch.randn((64, args.z_dim), device=device)
            X = G(sample_z).cpu()
            X = X.view(-1, args.img_channels, args.img_size, args.img_size)
            save_image(X, args.save_path, nrow=8, normalize=True, value_range=(-1, 1))
        else:
            sample_z = torch.randn((5, 1, args.z_dim), device=device)
            sample_z = sample_z.expand(5, args.n_classes, args.z_dim).reshape(5*args.n_classes, args.z_dim)
            sample_c = F.one_hot(torch.arange(0, args.n_classes, device=device).repeat(5, ), num_classes=args.n_classes)
            X = G(sample_z, sample_c).cpu()
            X = X.view(-1, args.img_channels, args.img_size, args.img_size)
            save_image(X, args.save_path, nrow=args.n_classes, normalize=True, value_range=(-1, 1))
    elif args.mode == 'walk':
        if not args.conditional:
            sample_z1 = torch.randn((5, args.z_dim), device=device)
            sample_z2 = torch.randn((5, args.z_dim), device=device)
            result = []
            for t in torch.linspace(0, 1, 15):
                result.append(G(sample_z1 * t + sample_z2 * (1 - t)).cpu())
            result = torch.stack(result, dim=1).reshape(5 * 15, args.img_channels, args.img_size, args.img_size)
            save_image(result, args.save_path, nrow=15, normalize=True, value_range=(-1, 1))
        else:
            sample_z1 = torch.randn((5, args.z_dim), device=device)
            sample_z2 = torch.randn((5, args.z_dim), device=device)
            sample_c1 = torch.randint(0, args.n_classes, (5, ), device=device)
            sample_c2 = torch.randint(0, args.n_classes, (5, ), device=device)
            sample_c1 = F.one_hot(sample_c1, num_classes=args.n_classes).view(-1, args.n_classes)
            sample_c2 = F.one_hot(sample_c2, num_classes=args.n_classes).view(-1, args.n_classes)
            result = []
            for t in torch.linspace(0, 1, 15):
                result.append(G(sample_z1 * t + sample_z2 * (1 - t), sample_c1 * t + sample_c2 * (1 - t)).cpu())
            result = torch.stack(result, dim=1).reshape(5 * 15, args.img_channels, args.img_size, args.img_size)
            save_image(result, args.save_path, nrow=15, normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser()
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
                                            ], required=True, help='choose a gan to generate fake images')
    parser.add_argument('--model_path', required=True, help='path to the saved model')
    parser.add_argument('--mode', choices=['random', 'walk'], required=True, help='generation mode. Options: random, walk')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the generated result')
    parser.add_argument('--cpu', action='store_true', help='use cpu instead of cuda')
    # Generator settings
    parser.add_argument('--z_dim', type=int, required=True, help='dimensionality of latent vector')
    parser.add_argument('--n_classes', type=int, help='number of classes for conditional generators')
    parser.add_argument('--data_dim', type=int, help='dimensionality of output data, for mlp-like generators')
    parser.add_argument('--img_size', type=int, help='size of output images, for cnn-like generators')
    parser.add_argument('--img_channels', type=int, help='number of channels of output images, for cnn-like generators')
    parser.add_argument('--conditional', action='store_true', help='whether the generator is conditional or not')
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()

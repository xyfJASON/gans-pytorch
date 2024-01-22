import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tqdm
import math
import argparse
from yacs.config import CfgNode as CN

import torch
import accelerate
from torchvision.utils import save_image

from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--seed', type=int, default=2022,
        help='Set random seed',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--n_samples', type=int, required=True,
        help='Number of samples',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--mode', type=str, default='sample', choices=['sample', 'traverse'],
        help='Sampling mode',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=128,
        help='Batch size on each process. Sample by batch is much faster',
    )
    return parser


def amortize(n_samples: int, batch_size: int):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


if __name__ == '__main__':
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD MODEL
    G = instantiate_from_config(cfg.G)
    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    G.load_state_dict(ckpt['G'])
    logger.info(f'Successfully load Generator from {args.weights}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    G = accelerator.prepare(G)
    G.eval()

    accelerator.wait_for_everyone()

    @torch.no_grad()
    def sample():
        idx = 0
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        batch_size = micro_batch * accelerator.num_processes
        for bs in tqdm.tqdm(
                amortize(args.n_samples, batch_size), desc='Sampling',
                disable=not accelerator.is_main_process,
        ):
            z = torch.randn((micro_batch, len(cfg.G.params.dim_mults), cfg.G.params.n_basis), device=device)
            x = torch.randn((micro_batch, cfg.G.params.noise_dim), device=device)
            samples = G(x, z).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                    idx += 1

    @torch.no_grad()
    @accelerator.on_main_process
    def sample_traverse():
        for l in range(len(cfg.G.params.dim_mults)):
            for d in range(cfg.G.params.n_basis):
                print(f'Sampling L{l}D{d}...')
                idx = 0
                z = torch.randn((args.n_samples, len(cfg.G.params.dim_mults), cfg.G.params.n_basis), device=device)
                z = z.unsqueeze(1).repeat(1, 9, 1, 1)
                z[:, :, l, d] = torch.linspace(-3, 3, 9, device=device)[None, :].repeat(args.n_samples, 1)
                z = z.view(-1, len(cfg.G.params.dim_mults), cfg.G.params.n_basis)
                x = torch.randn((args.n_samples, 1, cfg.G.params.noise_dim), device=device).repeat(1, 9, 1)
                x = x.view(-1, cfg.G.params.noise_dim)
                samples = accelerator.unwrap_model(G)(x, z).clamp(-1, 1)
                samples = samples.view(args.n_samples, 9, *samples.shape[1:])
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'L{l}D{d}_{idx}.png'), nrow=9)
                    idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    if args.mode == 'sample':
        sample()
    elif args.mode == 'traverse':
        sample_traverse()
    else:
        raise ValueError(f'Unknown sampling mode: {args.mode}')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')

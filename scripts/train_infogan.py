import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from utils.misc import get_time_str, check_freq
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(sort_keys=False),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    logger.info(f'=' * 30)

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    train_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.get('dataroot', None),
        img_size=cfg.data.get('img_size', None),
        split='train',
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')
    logger.info(f'=' * 30)

    # BUILD MODEL AND OPTIMIZERS
    G = instantiate_from_config(cfg.G)
    D = instantiate_from_config(cfg.D)
    cfg.train.optim_G.params.update({'params': G.parameters()})
    optimizer_G = instantiate_from_config(cfg.train.optim_G)
    cfg.train.optim_D.params.update({'params': D.parameters()})
    optimizer_D = instantiate_from_config(cfg.train.optim_D)
    step, best_fid = 0, math.inf

    def load_ckpt(ckpt_path: str):
        nonlocal step, best_fid
        # load models
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        G.load_state_dict(ckpt_model['G'])
        D.load_state_dict(ckpt_model['D'])
        logger.info(f'Successfully load models from {ckpt_path}')
        # load optimizers
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer_G.load_state_dict(ckpt_optimizer['optimizer_G'])
        optimizer_D.load_state_dict(ckpt_optimizer['optimizer_D'])
        logger.info(f'Successfully load optimizers from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1
        best_fid = ckpt_meta['best_fid']

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        accelerator.save(dict(
            G=accelerator.unwrap_model(G).state_dict(),
            D=accelerator.unwrap_model(D).state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save optimizers
        accelerator.save(dict(
            optimizer_G=optimizer_G.state_dict(),
            optimizer_D=optimizer_D.state_dict(),
        ), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(
            step=step, best_fid=best_fid,
        ), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')
        if best_fid != math.inf:
            logger.info(f'Best fid so far: {best_fid}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    G, D, optimizer_G, optimizer_D, train_loader = \
        accelerator.prepare(G, D, optimizer_G, optimizer_D, train_loader)  # type: ignore

    # DEFINE LOSS FUNCTION
    cfg.train.loss_fn.params.update({'discriminator': D})
    loss_fn = instantiate_from_config(cfg.train.loss_fn)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    accelerator.wait_for_everyone()

    def _discard_labels(x):
        if isinstance(x, (tuple, list)):
            return x[0]
        return x

    def run_step_D(X):
        optimizer_D.zero_grad()

        X = _discard_labels(X).float()
        z = torch.randn((X.shape[0], cfg.G.params.z_dim), device=device)
        c_disc, c_cont = None, None
        if cfg.G.params.dim_c_disc > 0:
            c_disc = torch.randint(0, cfg.G.params.dim_c_disc, (X.shape[0], ), device=device)
        if cfg.G.params.dim_c_cont > 0:
            c_cont = torch.randn((X.shape[0], cfg.G.params.dim_c_cont), device=device)

        fake = G(z, c_disc, c_cont).detach()
        loss = loss_fn.forward_D(fake, X)
        accelerator.backward(loss)
        optimizer_D.step()
        return dict(loss_D=loss.item(), lr_D=optimizer_D.param_groups[0]['lr'])

    def run_step_G(batch_size):
        optimizer_G.zero_grad()

        z = torch.randn((batch_size, cfg.G.params.z_dim), device=device)
        c_disc, c_cont = None, None
        if cfg.G.params.dim_c_disc > 0:
            c_disc = torch.randint(0, cfg.G.params.dim_c_disc, (batch_size, ), device=device)
        if cfg.G.params.dim_c_cont > 0:
            c_cont = torch.randn((batch_size, cfg.G.params.dim_c_cont), device=device)

        fake = G(z, c_disc, c_cont)
        loss = loss_fn.forward_G(fake)
        accelerator.backward(loss)
        optimizer_G.step()
        return dict(loss_G=loss.item(), lr_G=optimizer_G.param_groups[0]['lr'])

    def run_step_Q(batch_size):
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        z = torch.randn((batch_size, cfg.G.params.z_dim), device=device)
        c_disc, c_cont = None, None
        if cfg.G.params.dim_c_disc > 0:
            c_disc = torch.randint(0, cfg.G.params.dim_c_disc, (batch_size, ), device=device)
        if cfg.G.params.dim_c_cont > 0:
            c_cont = torch.randn((batch_size, cfg.G.params.dim_c_cont), device=device)

        fake = G(z, c_disc, c_cont)
        _, pred_logits, pred_mean = D(fake)
        loss_cond_disc = torch.tensor(0., device=device)
        loss_cond_cont = torch.tensor(0., device=device)
        if c_disc is not None and pred_logits is not None:
            loss_cond_disc = ce(pred_logits, c_disc)
        if c_cont is not None and pred_mean is not None:
            loss_cond_cont = mse(pred_mean, c_cont)
        loss = (cfg.train.lambda_cond_disc * loss_cond_disc +
                cfg.train.lambda_cond_cont * loss_cond_cont)
        accelerator.backward(loss)
        optimizer_G.step()
        optimizer_D.step()
        return dict(
            loss_info_disc=loss_cond_disc.item(),
            loss_info_cont=loss_cond_cont.item(),
        )

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        dirpath, filename = os.path.split(savepath)
        filename, ext = os.path.splitext(filename)
        unwrapped_G = accelerator.unwrap_model(G)
        if cfg.G.params.dim_c_disc > 0:
            nrow = min(cfg.G.params.dim_c_disc, 10)
            z = torch.randn((5, cfg.G.params.z_dim), device=device)
            z = z[:, None, :].repeat(1, nrow, 1).reshape(-1, cfg.G.params.z_dim)
            c_disc = torch.arange(nrow, device=device).repeat(5)
            c_cont = None
            if cfg.G.params.dim_c_cont > 0:
                c_cont = torch.randn((5, cfg.G.params.dim_c_cont), device=device)
                c_cont = c_cont[:, None, :].repeat(1, nrow, 1).reshape(-1, cfg.G.params.dim_c_cont)
            samples = unwrapped_G(z, c_disc, c_cont).cpu()
            samples = samples.view(-1, cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
            save_image(
                samples, os.path.join(dirpath, filename + '-disc' + ext),
                nrow=nrow, normalize=True, value_range=(-1, 1),
            )
        if cfg.G.params.dim_c_cont > 0:
            z = torch.randn((5, cfg.G.params.z_dim), device=device)
            z = z[:, None, :].repeat(1, 10, 1).reshape(-1, cfg.G.params.z_dim)
            c_cont = torch.randn((1, cfg.G.params.dim_c_cont), device=device).repeat(10, 1)
            c_cont[:, 0] = torch.linspace(-1., 1., 10, device=device)
            c_cont = c_cont.repeat(5, 1)
            c_disc = None
            if cfg.G.params.dim_c_disc > 0:
                c_disc = torch.randint(0, cfg.G.params.dim_c_disc, (5, ), device=device)
                c_disc = c_disc[:, None].repeat(1, 10).reshape(-1)
            samples = unwrapped_G(z, c_disc, c_cont).cpu()
            samples = samples.view(-1, cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
            save_image(
                samples, os.path.join(dirpath, filename + '-cont' + ext),
                nrow=10, normalize=True, value_range=(-1, 1),
            )

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        G.train(); D.train()

        # run multiple steps for discriminator
        for i in range(cfg.train.d_iters):
            batch = next(train_data_generator)
            train_status = run_step_D(batch)
            if i == cfg.train.d_iters - 1:
                status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        # run a step for generator
        train_status = run_step_G(batch_size_per_process)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        # run a step for Q
        train_status = run_step_Q(batch_size_per_process)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        G.eval(); D.eval()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(cfg.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    if best_fid != math.inf:
        logger.info(f'Best FID score: {best_fid}')
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()

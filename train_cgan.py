import os

import argparse
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

from losses import VanillaGANLoss
from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, instantiate_from_config


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
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
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
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    G, D, optimizer_G, optimizer_D, train_loader = \
        accelerator.prepare(G, D, optimizer_G, optimizer_D, train_loader)  # type: ignore

    # DEFINE LOSS
    vanilla_gan_loss = VanillaGANLoss(discriminator=D)

    accelerator.wait_for_everyone()

    def run_step_D(X, y):
        optimizer_D.zero_grad()
        X = X.float(); y = y.long()
        z = torch.randn((X.shape[0], cfg.G.params.z_dim), device=device)
        fake = G(z, y).detach()
        loss = vanilla_gan_loss.forward_D(fake, X, y)
        accelerator.backward(loss)
        optimizer_D.step()
        return dict(loss_D=loss.item(), lr_D=optimizer_D.param_groups[0]['lr'])

    def run_step_G(X, y):
        optimizer_G.zero_grad()
        X = X.float(); y = y.long()
        z = torch.randn((X.shape[0], cfg.G.params.z_dim), device=device)
        fake = G(z, y)
        loss = vanilla_gan_loss.forward_G(fake, y)
        accelerator.backward(loss)
        optimizer_G.step()
        return dict(loss_G=loss.item(), lr_G=optimizer_G.param_groups[0]['lr'])

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        unwrapped_G = accelerator.unwrap_model(G)
        all_samples = []
        for c in range(min(10, cfg.data.n_classes)):
            z = torch.randn((cfg.train.n_samples_per_class, cfg.G.params.z_dim), device=device)
            y = torch.full((cfg.train.n_samples_per_class, ), c, device=device, dtype=torch.long)
            samples = unwrapped_G(z, y).cpu()
            all_samples.append(samples)
        img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
        all_samples = torch.stack(all_samples, dim=1).view(-1, *img_shape)
        save_image(all_samples, savepath, nrow=min(10, cfg.data.n_classes), normalize=True, value_range=(-1, 1))

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        assert isinstance(batch, (list, tuple)) and len(batch) == 2
        # run a step
        G.train(); D.train()
        train_status = run_step_D(*batch)
        status_tracker.track_status('Train', train_status, step)
        if (step + 1) % cfg.train.d_iters == 0:
            train_status = run_step_G(*batch)
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

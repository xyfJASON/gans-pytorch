import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tqdm
import argparse
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

from models.ema import EMA
from utils.data import load_data
from utils.logger import get_logger
from utils.tracker import StatusTracker
from utils.misc import get_time_str, check_freq, set_seed
from utils.experiment import create_exp_dir, find_resume_checkpoint, instantiate_from_config, discard_label
from utils.distributed import init_distributed_mode, is_main_process, on_main_process, is_dist_avail_and_initialized
from utils.distributed import wait_for_everyone, cleanup, get_rank, get_world_size, get_local_rank


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-e', '--exp_dir', type=str, help='Path to the experiment directory. Default to be ./runs/exp-{current time}/')
    parser.add_argument('-r', '--resume', type=str, help='Resume from a checkpoint. Could be a path or `best` or `latest`')
    parser.add_argument('-cd', '--cover_dir', action='store_true', default=False, help='Cover the experiment directory if it exists')
    return parser


def main():
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE DISTRIBUTED MODE
    device = init_distributed_mode()
    print(f'Process {get_rank()} using device: {device}', flush=True)
    wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if is_main_process():
        create_exp_dir(
            exp_dir=exp_dir, conf_yaml=OmegaConf.to_yaml(conf), subdirs=['ckpt', 'samples'],
            time_str=args.time_str, exist_ok=args.resume is not None, cover_dir=args.cover_dir,
        )

    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True, is_main_process=is_main_process(),
    )

    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger, print_freq=conf.train.print_freq,
        tensorboard_dir=os.path.join(exp_dir, 'tensorboard'),
        is_main_process=is_main_process(),
    )

    # SET SEED
    set_seed(conf.seed + get_rank())
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {get_world_size()}')
    logger.info(f'Distributed mode: {is_dist_avail_and_initialized()}')
    wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert conf.train.batch_size % get_world_size() == 0
    bspp = conf.train.batch_size // get_world_size()
    train_set = load_data(conf.data, split='train')
    train_sampler = DistributedSampler(train_set, num_replicas=get_world_size(), rank=get_rank(), shuffle=True)
    train_loader = DataLoader(train_set, batch_size=bspp, sampler=train_sampler, drop_last=True, **conf.dataloader)
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {bspp}')
    logger.info(f'Total batch size: {conf.train.batch_size}')

    # BUILD MODEL AND OPTIMIZERS
    G = instantiate_from_config(conf.G).to(device)
    D = instantiate_from_config(conf.D).to(device)
    ema_G = EMA(G.parameters(), decay=conf.train.ema_decay)
    ema_G.to(device)
    optimizer_G = instantiate_from_config(conf.train.optim_G, params=G.parameters())
    optimizer_D = instantiate_from_config(conf.train.optim_D, params=D.parameters())
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Number of parameters of G: {sum(p.numel() for p in G.parameters()):,}')
    logger.info(f'Number of parameters of D: {sum(p.numel() for p in D.parameters()):,}')
    logger.info('=' * 50)

    # RESUME TRAINING
    step, epoch = 0, 0
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        # load models
        ckpt = torch.load(os.path.join(resume_path, 'model.pt'), map_location='cpu')
        G.load_state_dict(ckpt['G'])
        D.load_state_dict(ckpt['D'])
        logger.info(f'Successfully load models from {resume_path}')
        # load training states (optimizers, step, epoch, best_fid)
        ckpt = torch.load(os.path.join(resume_path, 'training_states.pt'), map_location='cpu')
        optimizer_G.load_state_dict(ckpt['optimizer_G'])
        optimizer_D.load_state_dict(ckpt['optimizer_D'])
        step = ckpt['step'] + 1
        epoch = ckpt['epoch']
        logger.info(f'Successfully load optimizers from {resume_path}')
        logger.info(f'Restart training at step {step}')
        del ckpt

    # DEFINE LOSS FUNCTION
    loss_fn = instantiate_from_config(conf.train.loss_fn, discriminator=D)

    # PREPARE FOR DISTRIBUTED TRAINING
    if is_dist_avail_and_initialized():
        G = DDP(G, device_ids=[get_local_rank()], output_device=get_local_rank())
        D = DDP(D, device_ids=[get_local_rank()], output_device=get_local_rank())
    G_wo_ddp = G.module if is_dist_avail_and_initialized() else G
    D_wo_ddp = D.module if is_dist_avail_and_initialized() else D
    wait_for_everyone()

    # TRAINING FUNCTIONS
    @on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        torch.save(dict(
            G=G_wo_ddp.state_dict(),
            D=D_wo_ddp.state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save ema model
        ema_G.apply_shadow(G.parameters())
        torch.save(dict(
            G=G_wo_ddp.state_dict(),
            D=D_wo_ddp.state_dict(),
        ), os.path.join(save_path, 'ema_model.pt'))
        ema_G.restore(G.parameters())
        # save ema
        torch.save(dict(
            ema_G=ema_G.state_dict(),
        ), os.path.join(save_path, 'ema.pt'))
        # save training states (optimizers, step, epoch, best_fid)
        torch.save(dict(
            optimizer_G=optimizer_G.state_dict(),
            optimizer_D=optimizer_D.state_dict(),
            step=step,
            epoch=epoch,
        ), os.path.join(save_path, 'training_states.pt'))

    def train_step(x):
        x = discard_label(x).float().to(device)
        z = torch.randn((x.shape[0], len(conf.G.params.dim_mults), conf.G.params.n_basis), device=device)
        noise = torch.randn((x.shape[0], conf.G.params.noise_dim), device=device)
        fake, ortho_reg = G(noise, z, return_ortho_reg=True)

        # zero the gradients of G
        optimizer_G.zero_grad()
        # forward and backward for G
        loss = loss_fn.forward_G(fake)
        loss = loss + conf.train.ortho_reg * ortho_reg
        loss.backward()
        # optimize G
        optimizer_G.step()
        ema_G.update(G.parameters())

        # zero the gradients of D
        optimizer_D.zero_grad()
        # forward and backward for D
        loss_D = loss_fn.forward_D(fake.detach(), x)
        loss_D.backward()
        # optimize D
        optimizer_D.step()

        return dict(
            loss_G=loss.item(),
            loss_D=loss_D.item(),
            ortho_reg=ortho_reg.item(),
            lr_G=optimizer_G.param_groups[0]['lr'],
            lr_D=optimizer_D.param_groups[0]['lr']
        )

    @on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        ema_G.apply_shadow(G.parameters())
        z = torch.randn((conf.train.n_samples, len(conf.G.params.dim_mults), conf.G.params.n_basis), device=device)
        x = torch.randn((conf.train.n_samples, conf.G.params.noise_dim), device=device)
        samples = G_wo_ddp(x, z).cpu()
        nrow = math.ceil(math.sqrt(conf.train.n_samples))
        samples = samples.view(-1, conf.data.img_channels, conf.data.img_size, conf.data.img_size)
        save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))
        ema_G.restore(G.parameters())

    # START TRAINING
    logger.info('Start training...')
    while step < conf.train.n_steps:
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        for _batch in tqdm.tqdm(train_loader, desc='Epoch', leave=False, disable=not is_main_process()):
            if step >= conf.train.n_steps:
                break
            # train a step
            G.train(); D.train()
            train_status = train_step(_batch)
            status_tracker.track_status('Train', train_status, step)
            wait_for_everyone()
            # validate
            G.eval(); D.eval()
            # save checkpoint
            if check_freq(conf.train.save_freq, step):
                save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
                wait_for_everyone()
            # sample from current model
            if check_freq(conf.train.sample_freq, step):
                sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
                wait_for_everyone()
            step += 1
        epoch += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    wait_for_everyone()

    # END OF TRAINING
    status_tracker.close()
    cleanup()
    logger.info('End of training')


if __name__ == '__main__':
    main()

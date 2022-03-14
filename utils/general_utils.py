import os
import shutil
import yaml
import torch


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def parse_config(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config['exp_name'] is None:
        raise ValueError('exp_name missing')

    device = torch.device('cuda' if config['use_gpu'] and torch.cuda.is_available() else 'cpu')
    print('using device:', device)

    logroot = os.path.join('runs', config['exp_name'])
    print('logroot:', logroot)
    if os.path.exists(logroot):
        if (config.get('resume_path') is None) or (os.path.realpath(config['resume_path']).find(os.path.realpath(logroot)) == -1):
            opt = input('logroot path already exists. Choose an option: cover/continue/[exit]: ')
            if opt == 'cover':
                shutil.rmtree(logroot)
            elif opt == 'continue':
                pass
            else:
                exit()

    if config.get('save_per_epochs') and not os.path.exists(os.path.join(logroot, 'ckpt')):
        os.makedirs(os.path.join(logroot, 'ckpt'))

    if not os.path.exists(os.path.join(logroot, 'tensorboard')):
        os.makedirs(os.path.join(logroot, 'tensorboard'))

    if not os.path.exists(os.path.join(logroot, 'config.yml')):
        shutil.copyfile(config_path, os.path.join(logroot, 'config.yml'))

    return config, device, logroot

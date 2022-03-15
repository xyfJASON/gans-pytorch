import argparse

from trainer import Trainer
from tester import Tester


def train(config_path):
    assert config_path
    trainer = Trainer(config_path)
    trainer.train()


def walk(model_path, z_dim, c_dim, data_dim, save_path):
    assert model_path and z_dim and data_dim and save_path
    tester = Tester(model_path, z_dim, c_dim, data_dim)
    tester.walk_in_latent_space(save_path)


def fix_z(model_path, z_dim, c_dim, data_dim, save_path):
    assert model_path and z_dim and c_dim and data_dim and save_path
    tester = Tester(model_path, z_dim, c_dim, data_dim)
    tester.fix_z(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'walk', 'fix_z'])
    parser.add_argument('--config_path', help='train')
    parser.add_argument('--model_path', help='walk / fix_z')
    parser.add_argument('--z_dim', type=int, help='walk / fix_z')
    parser.add_argument('--c_dim', type=int, help='walk / fix_z')
    parser.add_argument('--data_dim', type=int, help='walk / fix_z')
    parser.add_argument('--save_path', help='walk / fix_z')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.config_path)
    elif args.mode == 'walk':
        walk(args.model_path, args.z_dim, args.c_dim, args.data_dim, args.save_path)
    elif args.mode == 'fix_z':
        fix_z(args.model_path, args.z_dim, args.c_dim, args.data_dim, args.save_path)
    else:
        raise ValueError

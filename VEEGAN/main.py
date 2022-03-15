import argparse

from trainer import Trainer


def train(config_path):
    assert config_path
    trainer = Trainer(config_path)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'walk'])
    parser.add_argument('--config_path', help='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.config_path)
    else:
        raise ValueError

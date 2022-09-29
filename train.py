import argparse
import trainers


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
        trainer = trainers.GAN_Trainer(args.config_path)
    elif args.model == 'dcgan':
        trainer = trainers.DCGAN_Trainer(args.config_path)
    elif args.model == 'cgan':
        trainer = trainers.CGAN_Trainer(args.config_path)
    elif args.model == 'acgan':
        trainer = trainers.ACGAN_Trainer(args.config_path)
    elif args.model == 'wgan':
        trainer = trainers.WGAN_Trainer(args.config_path)
    elif args.model == 'wgan-gp':
        trainer = trainers.WGAN_GP_Trainer(args.config_path)
    elif args.model == 'sngan':
        trainer = trainers.SNGAN_Trainer(args.config_path)
    elif args.model == 'sngan-projection':
        trainer = trainers.SNGAN_projection_Trainer(args.config_path)
    elif args.model == 'lsgan':
        trainer = trainers.LSGAN_Trainer(args.config_path)
    elif args.model == 'sagan':
        trainer = trainers.SAGAN_Trainer(args.config_path)
    elif args.model == 'veegan':
        trainer = trainers.VEEGAN_Trainer(args.config_path)
    else:
        raise ValueError(f'{args.model} is not supported now.')

    trainer.train()

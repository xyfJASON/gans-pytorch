from omegaconf import DictConfig

import torchvision.datasets as dset
import torchvision.transforms as T


def load_data(conf: DictConfig, split='train'):
    """Keys in conf: 'name', 'dataroot', 'img_size'."""
    assert conf.get('name') is not None
    if conf.name.lower() not in ['ring8', 'grid25']:
        assert conf.get('dataroot') is not None
        assert conf.get('img_size') is not None

    if conf.name.lower() == 'ring8':
        from datasets.toy import Ring8
        dataset = Ring8()

    elif conf.name.lower() == 'grid25':
        from datasets.toy import Grid25
        dataset = Grid25()

    elif conf.name.lower() == 'mnist':
        transforms = T.Compose([
            T.Resize((conf.img_size, conf.img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        dataset = dset.MNIST(root=conf.dataroot, train=(split == 'train'), transform=transforms)

    elif conf.name.lower() in ['cifar10', 'cifar-10']:
        flip_p = 0.5 if split == 'train' else 0.0
        transforms = T.Compose([
            T.Resize((conf.img_size, conf.img_size)),
            T.RandomHorizontalFlip(flip_p),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = dset.CIFAR10(root=conf.dataroot, train=(split == 'train'), transform=transforms)

    elif conf.name.lower() in ['celeba-hq', 'celebahq']:
        from datasets.celeba_hq import CelebA_HQ
        flip_p = 0.5 if split == 'train' else 0.0
        transforms = T.Compose([
            T.Resize((conf.img_size, conf.img_size)),
            T.RandomHorizontalFlip(flip_p),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = CelebA_HQ(root=conf.dataroot, split=split, transform=transforms)

    elif conf.name.lower() == 'ffhq':
        from datasets.ffhq import FFHQ
        flip_p = 0.5 if split == 'train' else 0.0
        transforms = T.Compose([
            T.Resize((conf.img_size, conf.img_size)),
            T.RandomHorizontalFlip(flip_p),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = FFHQ(root=conf.dataroot, split=split, transform=transforms)

    elif conf.name.lower() == 'imagenet':
        from datasets.imagenet import ImageNet
        crop = T.RandomCrop if split == 'train' else T.CenterCrop
        flip_p = 0.5 if split == 'train' else 0.0
        transforms = T.Compose([
            T.Resize(conf.img_size),
            crop((conf.img_size, conf.img_size)),
            T.RandomHorizontalFlip(flip_p),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ImageNet(root=conf.dataroot, split=split, transform=transforms)

    else:
        raise ValueError(f'Unsupported dataset: {conf.name}')

    return dataset

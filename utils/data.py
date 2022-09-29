from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.datasets as dset

from datasets import Ring8, Grid25


def build_dataset(name, dataroot, img_size, split, transforms=None, subset_ids=None):
    if name == 'ring8':
        dataset = Ring8()
        data_dim = 2
        img_channels = None
        n_classes = None
    elif name == 'grid25':
        dataset = Grid25()
        data_dim = 2
        img_channels = None
        n_classes = None
    elif name == 'mnist':
        assert split in ['train', 'test'], f'{name} only has train/test split, get {split}.'
        if transforms is None:
            transforms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
        dataset = dset.MNIST(root=dataroot, train=(split == 'train'), transform=transforms)
        data_dim = img_size * img_size
        img_channels = 1
        n_classes = 10
    elif name == 'fashion_mnist':
        assert split in ['train', 'test'], f'{name} only has train/test split, get {split}.'
        if transforms is None:
            transforms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])
        dataset = dset.FashionMNIST(root=dataroot, train=(split == 'train'), transform=transforms)
        data_dim = img_size * img_size
        img_channels = 1
        n_classes = 10
    elif name == 'celeba':
        if transforms is None:
            transforms = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset = dset.CelebA(root=dataroot, split=split, transform=transforms)
        data_dim = img_size * img_size
        img_channels = 3
        n_classes = None
    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if subset_ids is not None:
        dataset = Subset(dataset, subset_ids)

    return dataset, data_dim, img_channels, n_classes


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False, is_ddp=False):
    if is_ddp:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader

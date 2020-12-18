from pathlib import Path

import torch
import torchvision
from torchvision import datasets, transforms


def get_dataset(name, train=True):
    if name == "EMNIST":
        return torch.utils.data.DataLoader(
            datasets.EMNIST(
                "~/Data",
                split="digits",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        # transforms.RandomAffine(0, (0.3, 0.3)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )

    elif name == "letters":
        return torch.utils.data.DataLoader(
            datasets.EMNIST(
                "~/Data",
                split="letters",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )

    elif name == "fashion":
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                "~/Data",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
        )

    elif name == "kmnist":
        return torch.utils.data.DataLoader(
            datasets.KMNIST(
                "~/Data",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.Pad(2),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                        transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                    ]
                ),
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
        )
    elif name == "CIFAR10":
        return torch.utils.data.DataLoader(
            datasets.CIFAR10(
                "~/Data",
                train=train,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
                        ),
                    ]
                ),
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True,
        )
    elif name == "FERG_expressions":
        return torch.utils.data.DataLoader(
            datasets.ImageFolder(
                "~/Data/FERG_expressions",
                transform=transforms.Compose(
                    [
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.2090, 0.1497, 0.1129), (0.2601, 0.1899, 0.1538)
                            # (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                        )
                    ]
                ),
                is_valid_file=lambda f: f not in ['/home/tillb/Data/FERG_expressions/surprise/bonnie_surprise_1389.png']
            ),
            batch_size=512,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
            drop_last=True
        )


def denormalize(name, image):
    if name in ['EMNIST', 'letters', 'fashion', 'kmnist']:
        mean = torch.tensor((0.5,), device=image.device).reshape(-1, 1, 1)
        std = torch.tensor((0.5,), device=image.device).reshape(-1, 1, 1)
    elif name == "CIFAR10":
        mean = torch.tensor((0.4914, 0.4822, 0.4465), device=image.device).reshape(-1, 1, 1)
        std = torch.tensor((0.247, 0.243, 0.261), device=image.device).reshape(-1, 1, 1)
    elif name == "FERG_expressions":
        mean = torch.tensor((0.2090, 0.1497, 0.1129), device=image.device).reshape(-1, 1, 1)
        std = torch.tensor((0.2601, 0.1899, 0.1538), device=image.device).reshape(-1, 1, 1)

    return image * std + mean


def tensors2image(dset, tensors, n=8):
    image = torchvision.utils.make_grid(tensors, n)
    return denormalize(dset, image.data).cpu().numpy()

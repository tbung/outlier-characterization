import torch
from torchvision import datasets, transforms


def get_dataset(name, train=True):
    if name == 'EMNIST':
        return torch.utils.data.DataLoader(
            datasets.EMNIST('~/Data', split='digits', train=train,
                            download=True,
                            transform=transforms.Compose([
                                transforms.Pad(2),
                                # transforms.RandomAffine(0, (0.5, 0.5)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                            ])),
            batch_size=512, shuffle=True, pin_memory=True, num_workers=8,
            drop_last=True
        )

    elif name == 'letters':
        return torch.utils.data.DataLoader(
            datasets.EMNIST('~/Data', split='letters', train=train, download=True,
                            transform=transforms.Compose([
                                transforms.Pad(2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                            ])),
            batch_size=512, shuffle=True, pin_memory=True, num_workers=8,
            drop_last=True
        )

    elif name == 'fashion':
        return torch.utils.data.DataLoader(
            datasets.FashionMNIST('~/Data', train=train, download=True,
                                  transform=transforms.Compose([
                                      transforms.Pad(2),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,)),
                                      transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                                      # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                  ])),
            batch_size=512, shuffle=True, pin_memory=True, num_workers=4,
            drop_last=True
        )

    elif name == 'kmnist':
        return torch.utils.data.DataLoader(
            datasets.KMNIST('~/Data', train=train, download=True,
                            transform=transforms.Compose([
                                transforms.Pad(2),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
                                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                            ])),
            batch_size=512, shuffle=True, pin_memory=True, num_workers=4,
            drop_last=True
        )

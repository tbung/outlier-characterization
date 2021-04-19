import torch
from torchvision import datasets, transforms

train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "~/Data",
        split="digits",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
            ]
        ),
    ),
    batch_size=512,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "~/Data",
        split="digits",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
            ]
        ),
    ),
    batch_size=512,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

letter_loader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "~/Data",
        split="letters",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.Lambda(lambda x: x.permute(0, 2, 1)),
            ]
        ),
    ),
    batch_size=512,
    shuffle=True,
    pin_memory=True,
    num_workers=4,
    drop_last=True,
)

import torch
from torchvision import datasets, transforms

# import config as c

train_loader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "~/Data",
        split="digits",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Pad(2),
                # transforms.RandomAffine(0, (0.5, 0.5)),
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


# train_loader = torch.utils.data.DataLoader(
#     datasets.FakeData(size=240000, image_size=(1, 32, 32),
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                       ])),
#     batch_size=c.batch_size, shuffle=True, pin_memory=True, num_workers=4,
#     drop_last=True
# )

# test_loader = torch.utils.data.DataLoader(
#     datasets.FakeData(size=40000, image_size=(1, 32, 32),
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                       ])),
#     batch_size=c.batch_size, shuffle=True, pin_memory=True, num_workers=4,
#     drop_last=True
# )


letter_loader = torch.utils.data.DataLoader(
    datasets.EMNIST(
        "~/Data",
        split="letters",
        train=True,
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

fashion_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        "~/Data",
        train=True,
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

kmnist_loader = torch.utils.data.DataLoader(
    datasets.KMNIST(
        "~/Data",
        train=True,
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

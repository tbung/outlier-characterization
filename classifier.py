from pathlib import Path
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

import data

checkpoints_path = Path("runs/classifier")
checkpoints_path.mkdir(exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "E": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    model = torchvision.models.VGG(
        make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs
    )
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


if __name__ == '__main__':
    train_loader = data.get_dataset("EMNIST")
    test_loader = data.get_dataset("EMNIST", train=False)

    model = vgg11(num_classes=10).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1e-2 ** (1e-1))

    for epoch in range(10):
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            acc = 0
            for x, y in tqdm(test_loader, leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)

                acc += (y == torch.argmax(out, dim=1)).float().mean()

            print(acc / len(test_loader))
            model.train()

        scheduler.step()

        torch.save(model.state_dict(), checkpoints_path / f"vgg11_{epoch}.pt")

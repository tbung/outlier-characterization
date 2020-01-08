import torch
import torch.nn as nn
import config as c


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(c.nch, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, c.ncl),
        )

    def forward(self, x):
        y = self.conv_layers(x)
        return self.fc_layers(y.reshape(x.size(0), -1))


class Generator(nn.Module):
    def __init__(self, conditional=False):
        super(Generator, self).__init__()
        self.conditional = conditional
        if conditional:
            self.img_features = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(c.nz, c.ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(c.ngf * 4),
                nn.ReLU(True),
            )

            self.label_features = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(c.ncl, c.ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(c.ngf * 4),
                nn.ReLU(True),
            )
        else:
            self.img_features = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(c.nz, c.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(c.ngf * 8),
                nn.ReLU(True),
            )

        self.main = nn.Sequential(
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(c.ngf * 8, c.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(c.ngf * 4, c.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            # nn.ConvTranspose2d(c.ngf * 2, c.ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(c.ngf),
            # nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(c.ngf * 2, c.nch, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, label):
        x = self.img_features(input)
        if self.conditional:
            y = self.label_features(label)
            x = torch.cat([x, y], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, conditional=False):
        super(Discriminator, self).__init__()
        self.conditional = conditional
        if conditional:
            self.img_features = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(c.nch, c.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

            self.label_features = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(c.ncl, c.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.img_features = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(c.nch, c.ndf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(c.ndf * 2, c.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(c.ndf * 4, c.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            # nn.Conv2d(c.ndf * 4, c.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(c.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(c.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

    def forward(self, input, label):
        x = self.img_features(input)
        if self.conditional:
            y = self.label_features(label)
            x = torch.cat([x, y], dim=1)
        return self.main(x)


class INN(nn.Module):
    # TODO
    pass

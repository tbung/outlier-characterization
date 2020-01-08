import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(c.nz, c.ngf),
            nn.ReLU(True),
            nn.Linear(c.ngf, c.ngf),
            nn.ReLU(True),
            nn.Linear(c.ngf, c.ngf),
            nn.ReLU(True),
            nn.Linear(c.ngf, 2),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, c.ndf),
            # nn.BatchNorm1d(c.ndf),
            nn.ReLU(True),
            nn.Linear(c.ndf, c.ndf),
            nn.BatchNorm1d(c.ndf),
            nn.ReLU(True),
            nn.Linear(c.ndf, c.ndf),
            nn.ReLU(True),
            nn.Linear(c.ndf, 1),
            nn.Sigmoid()
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs).view(-1)
        return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, c.ncf),
            nn.ReLU(True),
            nn.Linear(c.ncf, c.ncf),
            nn.ReLU(True),
            nn.Linear(c.ncf, 2)
        )

    def forward(self, inputs):
        return self.layers(inputs)


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, c.ngf), nn.ReLU(),
                         nn.Linear(c.ngf,  c_out))


def INN():
    nodes = [Ff.InputNode(2, name='input')]

    # Use a loop to produce a chain of coupling blocks
    for k in range(2):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                             name=F'coupling_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)

import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import numpy as np

from utils import config

c = config.Config()
c.load("./config/default.toml")


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(c.nch, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(6400, 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
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
    def __init__(self, c, conditional=False):
        super(Discriminator, self).__init__()
        self.conditional = conditional
        if conditional:
            self.img_features = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(c.nch, c.ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.2),
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
                nn.Dropout(0.2),
            )

        self.main = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(c.ndf * 2, c.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(c.ndf * 4, c.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(c.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
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


# def INN():
#     layer_types = {
#         'NICE': Fm.NICECouplingBlock,
#         'RNVP': Fm.RNVPCouplingBlock,
#         'GLOW': Fm.GLOWCouplingBlock,
#         'GIN': Fm.GINCouplingBlock,
#     }

#     cond_size = 10
#     cond_node = Ff.ConditionNode(cond_size)

#     def subnet_constructor(ch_in, ch_out):
#         return Fm.F_fully_connected(ch_in, ch_out, dropout=c.fc_dropout, internal_size=c.internal_width)

#     mod_args = {
#                 'subnet_constructor': subnet_constructor,
#     }

#     if c.couplig_type != 'NICE':
#         mod_args['clamp'] = c.clamping

#     nodes = [Ff.InputNode(c.nch, 32, 32, name='inp')]

#     nodes.append(Ff.Node([nodes[-1].out0], Fm.flattening_layer, {}, name='flatten'))

#     for i in range(c.n_blocks):
#         nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
#                              name=F'permute_{i}'))

#         nodes.append(Ff.Node(
#             [nodes[-1].out0], layer_types[c.couplig_type], mod_args, conditions=cond_node, name=F'fc_{i}'
#         ))

#     nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
#     nodes.append(cond_node)

#     return Ff.ReversibleGraphNet(nodes, verbose=False)


# def INN():
#     layer_types = {
#         'NICE': Fm.NICECouplingBlock,
#         'RNVP': Fm.RNVPCouplingBlock,
#         'GLOW': Fm.GLOWCouplingBlock,
#         'GIN': Fm.GINCouplingBlock,
#     }

#     ndim_x = 32 * 32

#     cond_size = 10
#     cond_nodes = [
#         Ff.ConditionNode(10, 16, 16),
#         Ff.ConditionNode(10, 8, 8),
#         Ff.ConditionNode(cond_size)
#     ]

#     def subnet_conv(ch_in, ch_out):
#         width = 32
#         return nn.Sequential(nn.Conv2d(ch_in, width, 3, padding=1), nn.ReLU(),
#                              nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
#                              nn.Conv2d(width, ch_out, 3, padding=1))

#     def subnet_conv_1x1(c_in, c_out):
#         width = 32
#         return nn.Sequential(nn.Conv2d(ch_in, width, 1), nn.ReLU(),
#                              nn.Conv2d(width, width, 1), nn.ReLU(),
#                              nn.Conv2d(width, ch_out, 1))

#     def subnet_fc(c_in, c_out):
#         width = 392
#         return nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
#                              nn.Linear(width, width), nn.ReLU(),
#                              nn.Linear(width,  c_out))

#     mod_args = {
#                 'subnet_constructor': subnet_conv,
#     }

#     if c.couplig_type != 'NICE':
#         mod_args['clamp'] = c.clamping

#     nodes = [Ff.InputNode(c.nch, c.img_width, c.img_width, name='inp')]

#     nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

#     for i in range(4):
#         nodes.append(Ff.Node(
#             [nodes[-1].out0], layer_types[c.couplig_type], mod_args,
#             conditions=cond_nodes[0], name=F'conv_high_res_{i}'
#         ))
#         nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
#                              name=F'permute_high_res_{i}'))

#     nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

#     mod_args = {
#                 'subnet_constructor': subnet_conv,
#     }

#     if c.couplig_type != 'NICE':
#         mod_args['clamp'] = c.clamping

#     for i in range(4):
#         if i % 2 == 0:
#             mod_args['subnet_constructor'] = subnet_conv_1x1
#         else:
#             mod_args['subnet_constructor'] = subnet_conv

#         nodes.append(Ff.Node(
#             [nodes[-1].out0], layer_types[c.couplig_type], mod_args,
#             conditions=cond_nodes[1], name=F'conv_low_res_{i}'
#         ))
#         nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
#                              name=F'permute_low_res_{i}'))

#     nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
#     # split_node = Ff.Node(nodes[-1],
#     #                      Fm.Split1D,
#     #                      {'split_size_or_sections':
#     #                       (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0},
#     #                      name='split')
#     # nodes.append(split_node)

#     mod_args = {
#                 'subnet_constructor': subnet_fc,
#     }

#     if c.couplig_type != 'NICE':
#         mod_args['clamp'] = c.clamping

#     for i in range(2):
#         nodes.append(Ff.Node(
#             [nodes[-1].out0], layer_types[c.couplig_type], mod_args,
#             conditions=cond_nodes[2], name=F'fc_{i}'
#         ))
#         nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
#                              name=F'permute_{i}'))

#     # nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
#     #                      Fm.Concat1d, {'dim': 0}, name='concat'))

#     nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
#     nodes.extend(cond_nodes)

#     return Ff.ReversibleGraphNet(nodes, verbose=False)

### PETER'S CODE AFTER THIS


def subnet_fc(c_in, c_out):
    width = 392
    subnet = nn.Sequential(
        nn.Linear(c_in, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        nn.Linear(width, c_out),
    )
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.0)
    subnet[-1].bias.data.fill_(0.0)
    return subnet


def subnet_conv1(c_in, c_out):
    width = 16
    subnet = nn.Sequential(
        nn.Conv2d(c_in, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, c_out, 3, padding=1),
    )
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.0)
    subnet[-1].bias.data.fill_(0.0)
    return subnet


def subnet_conv2(c_in, c_out):
    width = 32
    subnet = nn.Sequential(
        nn.Conv2d(c_in, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, c_out, 3, padding=1),
    )
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.0)
    subnet[-1].bias.data.fill_(0.0)
    return subnet


def construct_nodes():
    nodes = [Ff.InputNode(1, c.img_width, c.img_width, name="input")]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name="downsample1"))

    for k in range(4):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GINCouplingBlock,
                {"subnet_constructor": subnet_conv1, "clamp": 2.0},
                name=f"coupling_conv1_{k}",
            )
        )
        nodes.append(
            Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_conv1_{k}")
        )

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name="downsample2"))

    for k in range(4):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GINCouplingBlock,
                {"subnet_constructor": subnet_conv2, "clamp": 2.0},
                name=f"coupling_conv2_{k}",
            )
        )
        nodes.append(
            Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_conv2_{k}")
        )

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name="flatten"))

    for k in range(2):
        nodes.append(
            Ff.Node(
                nodes[-1],
                Fm.GINCouplingBlock,
                {"subnet_constructor": subnet_fc, "clamp": 2.0},
                name=f"coupling_fc_{k}",
            )
        )
        nodes.append(
            Ff.Node(nodes[-1], Fm.PermuteRandom, {"seed": k}, name=f"permute_fc_{k}")
        )

    nodes.append(Ff.OutputNode(nodes[-1], name="output"))
    return nodes


class INN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 10
        self.input_dims = (1, c.img_width, c.img_width)
        self.n_dims = int(np.prod(self.input_dims))
        print(self.n_dims)
        self.net = Ff.ReversibleGraphNet(construct_nodes(), verbose=False)
        self.mu = torch.nn.Parameter(
            torch.zeros(self.n_classes, self.n_dims).cuda()
        ).requires_grad_()
        self.log_sig = torch.nn.Parameter(
            torch.ones(self.n_classes, self.n_dims).cuda()
        ).requires_grad_()

    def forward(self, x, c, rev=False):
        self.logdet = 0.0
        x = self.net(x, rev=rev)
        self.logdet += self.net.log_jacobian(run_forward=False) / self.n_dims
        return x

    def jacobian(self, run_forward):
        return self.net.jacobian(run_forward=run_forward)

    def save(self, fname):
        torch.save({"inn": self.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data["inn"])

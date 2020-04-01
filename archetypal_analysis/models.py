import torch
import torch.nn as nn
import numpy as np
import FrEIA.framework as Ff
import FrEIA.modules as Fm


# define model
class Encoder(nn.Module):
    def __init__(self, z_fixed, latent_dim):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(64, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7*7*64, 100)
        )

        self.fc_A = nn.Linear(50, latent_dim + 1)
        self.fc_B = nn.Linear(50, latent_dim + 1)
        self.fc_sigma = nn.Linear(50, latent_dim)

        self.z_fixed = z_fixed

    def forward(self, x):
        # return Z_pred, mu, sigma, t
        latent_temp = self.layers(x)

        weights_, sigma_ = latent_temp[:, :50], latent_temp[:, 50:]

        A = torch.softmax(self.fc_A(weights_), dim=1)
        B = torch.softmax(self.fc_B(weights_).T, dim=1)

        sigma = torch.nn.functional.softplus(self.fc_sigma(sigma_))

        mu = A @ self.z_fixed

        z_pred = B @ mu

        t = torch.distributions.multivariate_normal.MultivariateNormal(mu, scale_tril=torch.diag_embed(sigma))

        return z_pred, mu, sigma, t


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.layers_pre = nn.Sequential(
            nn.Linear(latent_dim, 49),
            nn.ReLU(),
            nn.Linear(49, 49),
            nn.ReLU(),
        )
        self.layers = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 28*28),
            nn.ReLU(),
            nn.Linear(28*28, 28*28),
            nn.Sigmoid()
        )

        self.fc_sideinfo = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_hat = self.layers_pre(x)
        x_hat = x_hat.reshape(-1, 1, 7, 7)
        x_hat = self.layers(x_hat)
        x_hat = x_hat.reshape(-1, 1, 28, 28)
        # TODO: optionally trainable variance
        x_hat = torch.distributions.normal.Normal(x_hat, 1)
        x_hat = torch.distributions.independent.Independent(x_hat, 3)

        sideinfo = self.fc_sideinfo(x)

        return x_hat, sideinfo


class DeepAA(nn.Module):
    def __init__(self, z_fixed, latent_dim):
        super(DeepAA, self).__init__()

        self.encoder = Encoder(z_fixed, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, t):
        return self.decoder(t)


def INN(
    coupling_type='GLOW',
    nch=1,
    n_blocks=24,
    internal_width=512,
    clamping=1.5,
    fc_dropout=0.0
):
    layer_types = {
        'NICE': Fm.NICECouplingBlock,
        'RNVP': Fm.RNVPCouplingBlock,
        'GLOW': Fm.GLOWCouplingBlock,
        'GIN': Fm.GINCouplingBlock,
    }

    # cond_size = 10
    # cond_node = Ff.ConditionNode(cond_size)

    def subnet_constructor(ch_in, ch_out):
        return Fm.F_fully_connected(ch_in, ch_out, dropout=fc_dropout, internal_size=internal_width)

    mod_args = {
                'subnet_constructor': subnet_constructor,
    }

    if coupling_type != 'NICE':
        mod_args['clamp'] = clamping

    nodes = [Ff.InputNode(nch, 28, 28, name='inp')]

    nodes.append(Ff.Node([nodes[-1].out0], Fm.flattening_layer, {}, name='flatten'))

    for i in range(n_blocks):
        nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
                             name=F'permute_{i}'))

        nodes.append(Ff.Node(
            [nodes[-1].out0], layer_types[coupling_type], mod_args, name=F'fc_{i}',
            # conditions=cond_node
        ))

    nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
    # nodes.append(cond_node)

    return Ff.ReversibleGraphNet(nodes, verbose=False)


def subnet_fc(c_in, c_out):
    width = 392
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv1(c_in, c_out):
    width = 16
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def subnet_conv2(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_nodes():
    nodes = [Ff.InputNode(1, 28, 28, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_conv1, 'clamp':2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_conv1_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_conv2, 'clamp':2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1],
                             Fm.GLOWCouplingBlock,
                             {'subnet_constructor':subnet_fc, 'clamp':2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':k},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return nodes


class ConvINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_classes = 10
        self.input_dims = (1, 28, 28)
        self.n_dims = np.prod(self.input_dims)
        self.net = Ff.ReversibleGraphNet(construct_nodes(), verbose=False)
        self.mu = torch.nn.Parameter(torch.zeros(self.n_classes, self.n_dims).cuda()).requires_grad_()
        self.log_sig = torch.nn.Parameter(torch.ones(self.n_classes, self.n_dims).cuda()).requires_grad_()

    def forward(self, x, rev=False):
        self.logdet = 0.
        x = self.net(x, rev=rev)
        self.logdet += self.net.log_jacobian(run_forward=False) / self.n_dims
        return x

    def log_jacobian(self, run_forward):
        return self.net.log_jacobian(run_forward=run_forward)

    def save(self, fname):
        torch.save({'inn': self.state_dict()}, fname)

    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['inn'])


class INN_AA(nn.Module):
    def __init__(self, latent_dim, conv=True):
        super(INN_AA, self).__init__()

        self.layers_A = nn.Sequential(
            nn.Linear(28*28, latent_dim + 1),
            # nn.Softmax(dim=1)
        )
        self.layers_B = nn.Sequential(
            nn.Linear(28*28, latent_dim + 1),
            # nn.Softmax(dim=0)
        )
        self.layers_C = nn.Sequential(
            nn.Linear(latent_dim, 28*28),
            # nn.Softmax(dim=0)
        )

        if conv:
            self.inn = ConvINN()
        else:
            self.inn = INN()

    def forward(self, x):
        t = self.inn(x)
        A = self.layers_A(t)
        exp_A = torch.exp(A - A.max())
        A = exp_A / torch.sum(exp_A, dim=1, keepdim=True)  # ** (1/2)
        B = self.layers_B(t)
        exp_B = torch.exp(B - B.max())
        B = exp_B / torch.sum(exp_B, dim=1, keepdim=True)  # ** (1/2)

        return t, A, B.T

    def sample(self, A, z_fixed):
        A_ = torch.sin(A * np.pi * 2/3)
        t = self.layers_C(A_ @ z_fixed / np.sin(np.pi * 2/3))
        return self.inn(t, rev=True)

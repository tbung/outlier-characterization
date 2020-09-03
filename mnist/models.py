from functools import partial
import torch
import torch.nn as nn
import numpy as np

import FrEIA.framework as Ff
import FrEIA.modules as Fm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def construct_inn(img_width,
                  n_channels,
                  n_classes,
                  coupling_type,
                  clamping,
                  internal_widths,
                  conditional
                  ):
    '''Construct a invertible neural network'''

    layer_types = {
        'NICE': Fm.NICECouplingBlock,
        'RNVP': Fm.RNVPCouplingBlock,
        'GLOW': Fm.GLOWCouplingBlock,
        'GIN': Fm.GINCouplingBlock,
    }
    print(internal_widths)

    if conditional:
        cond_nodes = [
            Ff.ConditionNode(n_classes, img_width//2, img_width//2),
            Ff.ConditionNode(n_classes, img_width//4, img_width//4),
            Ff.ConditionNode(n_classes)
        ]
    else:
        cond_nodes = [None, None, None]

    def subnet_conv(ch_in, ch_out, width):
        subnet = nn.Sequential(nn.Conv2d(ch_in, width, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(width, ch_out, 3, padding=1))
        for l in subnet:
            if isinstance(l, nn.Conv2d):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
        return subnet

    def subnet_conv_1x1(ch_in, ch_out, width):
        subnet = nn.Sequential(nn.Conv2d(ch_in, width, 1), nn.ReLU(),
                                nn.Conv2d(width, width, 1), nn.ReLU(),
                                nn.Conv2d(width, ch_out, 1))
        for l in subnet:
            if isinstance(l, nn.Conv2d):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
        return subnet

    def subnet_fc(ch_in, ch_out):
        width = 392
        subnet = nn.Sequential(nn.Linear(ch_in, width), nn.ReLU(),
                               nn.Linear(width, width), nn.ReLU(),
                               nn.Linear(width, ch_out))
        for l in subnet:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
        return subnet

    mod_args = {
                'subnet_constructor': partial(subnet_conv, width=internal_widths[0]),
    }

    if coupling_type != 'NICE':
        mod_args['clamp'] = clamping

    nodes = [Ff.InputNode(n_channels, img_width, img_width, name='inp')]

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    for i in range(4):
        nodes.append(Ff.Node(
            [nodes[-1].out0], layer_types[coupling_type], mod_args,
            conditions=cond_nodes[0], name=F'conv_high_res_{i}'
        ))
        nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
                             name=F'permute_high_res_{i}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    mod_args = {
                'subnet_constructor': partial(subnet_conv, width=internal_widths[1]),
    }

    if coupling_type != 'NICE':
        mod_args['clamp'] = clamping

    for i in range(4):
        if i % 2 == 0:
            mod_args['subnet_constructor'] = partial(subnet_conv_1x1,
                                                     width=internal_widths[1])
        else:
            mod_args['subnet_constructor'] = partial(subnet_conv, width=internal_widths[1])

        nodes.append(Ff.Node(
            [nodes[-1].out0], layer_types[coupling_type], mod_args,
            conditions=cond_nodes[1], name=F'conv_low_res_{i}'
        ))
        nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
                             name=F'permute_low_res_{i}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
    # split_node = Ff.Node(nodes[-1],
    #                      Fm.Split1D,
    #                      {'split_size_or_sections':
    #                       (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0},
    #                      name='split')
    # nodes.append(split_node)

    mod_args = {
                'subnet_constructor': subnet_fc,
    }

    if coupling_type != 'NICE':
        mod_args['clamp'] = clamping

    for i in range(2):
        nodes.append(Ff.Node(
            [nodes[-1].out0], layer_types[coupling_type], mod_args,
            conditions=cond_nodes[2], name=F'fc_{i}'
        ))
        nodes.append(Ff.Node([nodes[-1].out0], Fm.permute_layer, {'seed': i},
                             name=F'permute_{i}'))

    # nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
    #                      Fm.Concat1d, {'dim': 0}, name='concat'))

    nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
    if conditional:
        nodes.extend(cond_nodes)

    return Ff.ReversibleGraphNet(nodes, verbose=False)


class INN(nn.Module):
    def __init__(self,
                 img_width=32,
                 n_channels=1,
                 n_classes=10,
                 coupling_type='GLOW',
                 conditional=True,
                 internal_width1=32,
                 internal_width2=32,
                 clamping=1.5,
                 load_inn=False,
                 latent_dist='normal',
                 use_min_likelihood=False,
                 lambda_mll=1,
                 **kwargs
                 ):
        super(INN, self).__init__()

        self.img_width = img_width
        self.ncl = n_classes
        self.latent_dist = latent_dist
        self.use_min_likelihood = use_min_likelihood
        self.lambda_mll = lambda_mll
        self.conditional = conditional

        self.inn = construct_inn(img_width, n_channels, n_classes,
                                 coupling_type, clamping, [internal_width1,
                                 internal_width2], conditional)

        if load_inn:
            self.inn.load_state_dict(
                dict(filter(
                    lambda x: 'tmp' not in x[0], map(
                        lambda x: (x[0].replace('inn.', ''), x[1]), torch.load(load_inn).items()
                    ))))

    def forward(self, x, cond=None):
        if not self.conditional:
            cond = None
        t = self.inn(x, cond)
        return t

    def sample(self, t, cond=None):
        if not self.conditional:
            cond = None
        return self.inn(t, cond, rev=True)

    def labels2condition(self, labels):
        '''Convert a tensor of class labels to conditions for this network
        '''
        cond_tensor = torch.zeros(labels.size(0), self.ncl, device=device)
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.)

        fill = torch.zeros(
            (self.ncl, self.ncl, self.img_width, self.img_width),
            device=device)
        for i in range(self.ncl):
            fill[i, i, :, :] = 1

        cond = [
            fill[:, :, :self.img_width//2, :self.img_width//2][labels],
            fill[:, :, :self.img_width//4, :self.img_width//4][labels],
            cond_tensor
        ]
        return cond

    def negative_log_likelihood(self, latent, labels):
        '''Computes the log-likelihood of samples

        Negative log-likelihood ofr in-distribution samples and positive
        log-likelihood of out-of-distribution samples is returned.
        '''

        if self.latent_dist == 'mixture':
            # neg_log_likeli = torch.mean(
            #     (output - generator_in.mu[targets]
            #      / torch.exp(generator_in.log_sig[targets]))**2
            #     / 2 + generator_in.log_sig[targets], dim=1)

            mean = torch.zeros(self.ncl, latent.shape[1], dtype=torch.float,
                               device=device)
            var = torch.zeros(self.ncl, latent.shape[1], dtype=torch.float,
                              device=device)
            for i in range(self.ncl):
                mean[i] = latent[labels == i].mean(dim=0)
                var[i] = latent[labels == i].var(dim=0)

            neg_log_likeli = torch.mean(
                (latent - mean[labels])**2
                / var[labels]
                / 2 + 0.5*torch.log(var[labels]), dim=1)

        elif self.latent_dist == 'normal':
            zz = torch.sum(latent**2, dim=1)
            jac = self.inn.jacobian(run_forward=False)

            neg_log_likeli = 0.5 * zz - jac
        else:
            raise Exception('Unknown latent distribution')

        nll = torch.mean(neg_log_likeli)

        return nll

    def compute_losses(self, samples, labels, ood_samples=None):
        '''Computes the losses for this model
        '''

        losses = {}
        cond = self.labels2condition(labels)
        latent = self(samples, cond)
        losses["NLL"] = self.negative_log_likelihood(latent, labels)
        if self.use_min_likelihood:
            latent = self(ood_samples, cond)
            losses["OoD LL"] = self.negative_log_likelihood(latent,
                                                            labels)

        return losses

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            dict(filter(
                lambda x: 'tmp' not in x[0], torch.load(path).items()
            )))


def build_z_simplex(latent_dim, use_inlier=False):
    '''Return vertices of a simplex in dimension param:laten_dim'''
    z_fixed_t = np.zeros([latent_dim, latent_dim + (2 if use_inlier else 1)])

    for k in range(0, latent_dim):
        s = 0.0
        for i in range(0, k):
            s = s + z_fixed_t[i, k] ** 2

        z_fixed_t[k, k] = np.sqrt(1.0 - s)

        for j in range(k + 1, latent_dim + 1):
            s = 0.0
            for i in range(0, k):
                s = s + z_fixed_t[i, k] * z_fixed_t[i, j]

            z_fixed_t[k, j] = (-1.0 / float(latent_dim) - s) / z_fixed_t[k, k]
            z_fixed = np.transpose(z_fixed_t)
    return torch.tensor(z_fixed, device=device, dtype=torch.float,
                        requires_grad=False)


def build_z_from_letters(latent_dim):
    # TODO
    pass


class INN_AA(nn.Module):
    def __init__(self,
                 latent_dim=2,
                 img_width=32,
                 n_channels=1,
                 n_classes=10,
                 coupling_type="GLOW",
                 clamping=1.5,
                 load_inn=False,
                 latent_dist='normal',
                 weight_norm_exp=1,
                 weight_norm_constraint=1,
                 interpolation='linear',
                 fix_inn=False,
                 fix_z_arch=True,
                 use_proto_z=False,
                 z_from_similar=False,
                 lambda_at=10,
                 lambda_recon=1,
                 lambda_class=1,
                 lambda_proto=100,
                 internal_width1=32,
                 internal_width2=32,
                 **kwargs
                 ):
        super(INN_AA, self).__init__()

        self.weight_norm_exp = weight_norm_exp
        self.weight_norm_constraint = weight_norm_constraint
        self.interpolation = interpolation
        self.lambda_at = lambda_at
        self.lambda_recon = lambda_recon
        self.lambda_class = lambda_class
        self.lambda_proto = lambda_proto
        self.use_proto_z = use_proto_z
        self.fix_inn = fix_inn


        # TODO: Handle other cases
        if z_from_similar:
            self.z_arch = build_z_from_letters(latent_dim)
        else:
            self.z_arch = build_z_simplex(latent_dim, use_proto_z)

        if not fix_z_arch:
            self.z_arch = nn.Parameter(self.z_arch)

        n_archs = latent_dim + (2 if use_proto_z else 1)

        self.layers_A = nn.Sequential(
            nn.Linear(32*32, n_archs),
            # nn.Softmax(dim=1)
        )
        self.layers_B = nn.Sequential(
            nn.Linear(32*32, n_archs),
            # nn.Softmax(dim=0)
        )
        self.layers_C = nn.Sequential(
            nn.Linear(latent_dim, 32*32),
            # nn.Softmax(dim=0)
        )

        self.layers_sideinfo = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            # nn.Softmax(dim=1)
        )

        self.inn = INN(
            img_width=img_width,
            n_channels=n_channels,
            n_classes=n_classes,
            coupling_type=coupling_type,
            clamping=clamping,
            load_inn=load_inn,
            latent_dist=latent_dist,
            internal_width1=internal_width1,
            internal_width2=internal_width2,
        )

        if fix_inn:
            for param in self.inn.parameters():
                param.requires_grad = False

    def forward(self, x, cond=None):
        t = self.inn(x, cond)
        A = self.layers_A(t)
        exp_A = torch.exp(A - A.max())
        A = (
            exp_A / torch.sum(exp_A**self.weight_norm_exp, dim=1, keepdim=True)
            ** (1/self.weight_norm_exp)
        )
        A = self.weight_norm_constraint * A
        B = self.layers_B(t)
        exp_B = torch.exp(B - B.max())
        B = (
            exp_B / torch.sum(exp_B**self.weight_norm_exp, dim=1, keepdim=True)
            ** (1/self.weight_norm_exp)
        )
        B = self.weight_norm_constraint * B

        return t, A, B.T

    def sample(self, A, cond=None):
        if self.interpolation == 'linear':
            t = self.layers_C(A @ self.z_arch)
            sideinfo = self.layers_sideinfo(A @ self.z_arch)
        elif self.interpolation == 'slerp':
            A_ = torch.sin(A * np.pi * 2/3)
            t = self.layers_C(A_ @ self.z_arch / np.sin(np.pi * 2/3))
            sideinfo = self.layers_sideinfo(
                A_ @ self.z_arch / np.sin(np.pi * 2 / 3)
            )
        return self.inn.sample(t, cond), sideinfo

    def compute_losses(self, samples, labels):
        losses = {}
        cond = self.labels2condition(labels)
        t, A, B = self(samples, cond)

        recreated, sideinfo = self.sample(A, cond)

        sample_latent_mean = A @ self.z_arch
        at_loss = torch.mean(torch.norm(
            B @ sample_latent_mean - self.z_arch, dim=1
        ))
        recon_loss = torch.norm(recreated - samples)
        class_loss = nn.functional.cross_entropy(sideinfo, labels)

        if not self.fix_inn:
            neg_log_likeli = self.inn.negative_log_likelihood(t, labels)
            losses['NLL'] = neg_log_likeli
        losses['AT'] = self.lambda_at * at_loss
        losses['Recon'] = self.lambda_recon * recon_loss
        losses['Class'] = self.lambda_class * class_loss
        if self.use_proto_z:
            proto_loss = torch.sum(
                (self.z_arch[-1]
                 - torch.mean(sample_latent_mean, dim=0))**2
            )
            losses['Proto'] = self.lambda_proto * proto_loss

        return losses

    def labels2condition(self, labels):
        return self.inn.labels2condition(labels)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            dict(filter(
                lambda x: 'tmp' not in x[0], torch.load(path).items()
            )))


def get_model(name):
    if name == 'INN':
        return INN
    elif name == 'INN_AA':
        return INN_AA
    elif name == 'DCGAN':
        raise NotImplementedError()
    else:
        raise Exception(f'Model type {name} is not known')

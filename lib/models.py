from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import inn_architecture
import configparser
import os


device = "cuda" if torch.cuda.is_available() else "cpu"


def construct_inn(
    img_width,
    n_channels,
    n_classes,
    coupling_type,
    clamping,
    internal_widths,
    conditional,
    n_conv_high_res,
    n_conv_low_res,
    n_fc,
):
    """Construct a invertible neural network"""

    layer_types = {
        "NICE": Fm.NICECouplingBlock,
        "RNVP": Fm.RNVPCouplingBlock,
        "GLOW": Fm.GLOWCouplingBlock,
        "GIN": Fm.GINCouplingBlock,
    }
    # print(internal_widths)

    if conditional:
        cond_nodes = [
            Ff.ConditionNode(n_classes, img_width // 2, img_width // 2),
            Ff.ConditionNode(n_classes, img_width // 4, img_width // 4),
            Ff.ConditionNode(n_classes),
        ]
    else:
        cond_nodes = [[], [], []]

    def subnet_conv(ch_in, ch_out, width):
        subnet = nn.Sequential(
            nn.Conv2d(ch_in, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, ch_out, 3, padding=1),
        )
        for l in subnet:
            if isinstance(l, nn.Conv2d):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.0)
        subnet[-1].bias.data.fill_(0.0)
        return subnet

    def subnet_conv_1x1(ch_in, ch_out, width):
        subnet = nn.Sequential(
            nn.Conv2d(ch_in, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, ch_out, 1),
        )
        for l in subnet:
            if isinstance(l, nn.Conv2d):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.0)
        subnet[-1].bias.data.fill_(0.0)
        return subnet

    def subnet_fc(ch_in, ch_out):
        width = 392
        subnet = nn.Sequential(
            nn.Linear(ch_in, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, ch_out),
        )
        for l in subnet:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
        subnet[-1].weight.data.fill_(0.0)
        subnet[-1].bias.data.fill_(0.0)
        return subnet

    mod_args = {
        "subnet_constructor": partial(subnet_conv, width=internal_widths[0]),
    }

    if coupling_type != "NICE":
        mod_args["clamp"] = clamping

    nodes = [Ff.InputNode(n_channels, img_width, img_width, name="inp")]

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    for i in range(n_conv_high_res):
        nodes.append(
            Ff.Node(
                [nodes[-1].out0],
                layer_types[coupling_type],
                mod_args,
                conditions=cond_nodes[0],
                name=f"conv_high_res_{i}",
            )
        )
        nodes.append(
            Ff.Node(
                [nodes[-1].out0],
                Fm.permute_layer,
                {"seed": i},
                name=f"permute_high_res_{i}",
            )
        )

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

    mod_args = {
        "subnet_constructor": partial(subnet_conv, width=internal_widths[1]),
    }

    if coupling_type != "NICE":
        mod_args["clamp"] = clamping

    for i in range(n_conv_low_res):
        if i % 2 == 0:
            mod_args["subnet_constructor"] = partial(
                subnet_conv_1x1, width=internal_widths[1]
            )
        else:
            mod_args["subnet_constructor"] = partial(
                subnet_conv, width=internal_widths[1]
            )

        nodes.append(
            Ff.Node(
                [nodes[-1].out0],
                layer_types[coupling_type],
                mod_args,
                conditions=cond_nodes[1],
                name=f"conv_low_res_{i}",
            )
        )
        nodes.append(
            Ff.Node(
                [nodes[-1].out0],
                Fm.permute_layer,
                {"seed": i},
                name=f"permute_low_res_{i}",
            )
        )

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name="flatten"))
    # split_node = Ff.Node(nodes[-1],
    #                      Fm.Split1D,
    #                      {'split_size_or_sections':
    #                       (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0},
    #                      name='split')
    # nodes.append(split_node)

    mod_args = {
        "subnet_constructor": subnet_fc,
    }

    if coupling_type != "NICE":
        mod_args["clamp"] = clamping

    for i in range(n_fc):
        nodes.append(
            Ff.Node(
                [nodes[-1].out0],
                layer_types[coupling_type],
                mod_args,
                conditions=cond_nodes[2],
                name=f"fc_{i}",
            )
        )
        nodes.append(
            Ff.Node(
                [nodes[-1].out0], Fm.permute_layer, {"seed": i}, name=f"permute_{i}"
            )
        )

    if n_fc == 0:
        del cond_nodes[2]

    # nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
    #                      Fm.Concat1d, {'dim': 0}, name='concat'))

    nodes.append(Ff.OutputNode([nodes[-1].out0], name="out"))
    if conditional:
        nodes.extend(cond_nodes)

    return Ff.ReversibleGraphNet(nodes, verbose=False)


class INN(nn.Module):
    def __init__(
        self,
        img_width=32,
        n_channels=1,
        n_classes=10,
        coupling_type="GLOW",
        conditional=True,
        internal_width1=32,
        internal_width2=32,
        clamping=1.5,
        load_inn=False,
        latent_dist="normal",
        use_min_likelihood=False,
        lambda_mll=1,
        n_conv_high_res=4,
        n_conv_low_res=4,
        n_fc=2,
        **kwargs,
    ):
        super(INN, self).__init__()

        self.img_width = img_width
        self.latent_dim = img_width * img_width * n_channels
        self.ncl = n_classes
        self.latent_dist = latent_dist
        self.use_min_likelihood = use_min_likelihood
        self.lambda_mll = lambda_mll
        self.conditional = conditional
        self.n_fc = n_fc

        self.inn = construct_inn(
            img_width,
            n_channels,
            n_classes,
            coupling_type,
            clamping,
            [internal_width1, internal_width2],
            conditional,
            n_conv_high_res,
            n_conv_low_res,
            n_fc,
        )

        if load_inn:
            self.inn.load_state_dict(
                dict(
                    filter(
                        lambda x: "tmp" not in x[0],
                        map(
                            lambda x: (x[0].replace("inn.", ""), x[1]),
                            torch.load(load_inn).items(),
                        ),
                    )
                )
            )

    def forward(self, x, cond=None):
        if not self.conditional:
            cond = None
        elif self.n_fc == 0:
            cond = cond[:2]
        t = self.inn(x, cond)
        return t

    def sample(self, t, cond=None):
        if not self.conditional:
            cond = None
        elif self.n_fc == 0:
            cond = cond[:2]
        return self.inn(t, cond, rev=True)

    def labels2condition(self, labels):
        """Convert a tensor of class labels to conditions for this network"""
        cond_tensor = torch.zeros(labels.size(0), self.ncl, device=device)
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.0)

        fill = torch.zeros(
            (self.ncl, self.ncl, self.img_width, self.img_width), device=device
        )
        for i in range(self.ncl):
            fill[i, i, :, :] = 1

        cond = [
            fill[:, :, : self.img_width // 2, : self.img_width // 2][labels],
            fill[:, :, : self.img_width // 4, : self.img_width // 4][labels],
            cond_tensor,
        ]
        return cond

    def negative_log_likelihood(self, latent, labels):
        """Computes the log-likelihood of samples

        Negative log-likelihood ofr in-distribution samples and positive
        log-likelihood of out-of-distribution samples is returned.
        """

        if self.latent_dist == "mixture":
            # neg_log_likeli = torch.mean(
            #     (output - generator_in.mu[targets]
            #      / torch.exp(generator_in.log_sig[targets]))**2
            #     / 2 + generator_in.log_sig[targets], dim=1)

            mean = torch.zeros(
                self.ncl, latent.shape[1], dtype=torch.float, device=device
            )
            var = torch.zeros(
                self.ncl, latent.shape[1], dtype=torch.float, device=device
            )
            for i in range(self.ncl):
                mean[i] = latent[labels == i].mean(dim=0)
                var[i] = latent[labels == i].var(dim=0)

            neg_log_likeli = torch.mean(
                (latent - mean[labels]) ** 2 / var[labels] / 2
                + 0.5 * torch.log(var[labels]),
                dim=1,
            )

        elif self.latent_dist == "normal":
            zz = torch.sum(latent ** 2, dim=1)
            jac = self.inn.log_jacobian(run_forward=False)

            neg_log_likeli = 0.5 * zz - jac
        else:
            raise Exception("Unknown latent distribution")

        nll = torch.mean(neg_log_likeli)

        return nll

    def compute_losses(self, samples, labels, ood_samples=None):
        """Computes the losses for this model"""

        losses = {}
        cond = self.labels2condition(labels)
        latent = self(samples, cond)
        losses["NLL"] = self.negative_log_likelihood(latent, labels)
        if self.use_min_likelihood:
            latent = self(ood_samples, cond)
            losses["OoD LL"] = self.negative_log_likelihood(latent, labels)

        return losses

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            dict(filter(lambda x: "tmp" not in x[0], torch.load(path).items()))
        )


def build_z_simplex(
    latent_dim, use_inlier=False, requires_grad=False, z_per_class=False, n_classes=1
):
    """Return vertices of a simplex in dimension param:laten_dim"""
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

    for n in range(latent_dim):
        for i in range(n):
            R = np.eye(latent_dim)

            e = 1e-64 if z_fixed[n, i + 1] == 0 else 0
            a = np.arctan(z_fixed[n, i] / (z_fixed[n, i + 1] + e))

            R[i, i] = np.cos(a)
            R[i, i + 1] = np.sin(a)
            R[i + 1, i] = -np.sin(a)
            R[i + 1, i + 1] = np.cos(a)

            z_fixed = z_fixed @ R

    z_fixed = torch.tensor(
        z_fixed, device=device, dtype=torch.float, requires_grad=requires_grad
    )

    if z_per_class:
        return z_fixed.repeat(n_classes, 1, 1)
    else:
        return z_fixed


def build_z_from_letters(latent_dim):
    # TODO
    pass


class INN_AA(nn.Module):
    def __init__(
        self,
        latent_dim=2,
        img_width=32,
        n_channels=1,
        n_classes=10,
        coupling_type="GLOW",
        conditional=True,
        clamping=1.5,
        load_inn=False,
        latent_dist="normal",
        weight_norm_exp=1,
        weight_norm_constraint=1,
        interpolation="linear",
        fix_inn=False,
        fix_z_arch=True,
        use_proto_z=False,
        z_from_similar=False,
        z_per_class=False,
        lambda_nll=1,
        lambda_at=10,
        lambda_recon=1,
        lambda_class=1,
        lambda_proto=100,
        lambda_jac=0,
        internal_width1=32,
        internal_width2=32,
        n_conv_high_res=4,
        n_conv_low_res=4,
        n_fc=2,
        aa_allow_negative=False,
        aa_weights_min=0.0,
        aa_weights_max=1.0,
        aa_weights_noise=0,
        load_ib_inn=False,
        nullspace_split=False,
        aa_bias=True,
        soft_constraint=False,
        lambda_constraint=1,
        **kwargs,
    ):
        super(INN_AA, self).__init__()

        self.weight_norm_exp = weight_norm_exp
        self.weight_norm_constraint = weight_norm_constraint
        self.interpolation = interpolation
        self.lambda_nll = lambda_nll
        self.lambda_at = lambda_at
        self.lambda_recon = lambda_recon
        self.lambda_class = lambda_class
        self.lambda_proto = lambda_proto
        self.lambda_jac = lambda_jac
        self.use_proto_z = use_proto_z
        self.fix_inn = fix_inn
        self.z_per_class = z_per_class
        self.latent_dim = latent_dim
        self.aa_allow_negative = aa_allow_negative
        self.aa_weights_min = aa_weights_min
        self.aa_weights_max = aa_weights_max
        self.aa_weights_noise = aa_weights_noise
        self.nullspace_split = nullspace_split
        self.soft_constraint = soft_constraint
        self.lambda_constraint = lambda_constraint

        # TODO: Handle other cases
        if z_from_similar:
            self.z_arch = build_z_from_letters(latent_dim)
        else:
            self.z_arch = build_z_simplex(
                latent_dim,
                use_proto_z,
                requires_grad=not fix_z_arch,
                z_per_class=z_per_class,
                n_classes=n_classes,
            )

        if not fix_z_arch:
            self.z_arch = nn.Parameter(self.z_arch, requires_grad=True)

        n_archs = latent_dim + (2 if use_proto_z else 1)

        self.layers_A = nn.Sequential(
            nn.Linear(n_channels * img_width * img_width, n_archs, aa_bias),
            # nn.Softmax(dim=1)
        )
        self.layers_B = nn.Sequential(
            nn.Linear(n_channels * img_width * img_width, n_archs, aa_bias),
            # nn.Softmax(dim=0)
        )
        self.layers_C = nn.Sequential(
            nn.Linear(latent_dim, n_channels * img_width * img_width, aa_bias),
            # nn.Softmax(dim=0)
        )

        self.layers_sideinfo = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
            # nn.Softmax(dim=1)
        )

        if load_ib_inn:
            self.inn = IB_INN(Path(load_ib_inn))
            self.inn.load(Path(load_ib_inn) / "model.pt")
        else:
            self.inn = INN(
                img_width=img_width,
                n_channels=n_channels,
                n_classes=n_classes,
                coupling_type=coupling_type,
                conditional=True,
                clamping=clamping,
                load_inn=load_inn,
                latent_dist=latent_dist,
                internal_width1=internal_width1,
                internal_width2=internal_width2,
                n_conv_high_res=n_conv_high_res,
                n_conv_low_res=n_conv_low_res,
                n_fc=n_fc,
            )

        if fix_inn:
            for param in self.inn.parameters():
                param.requires_grad = False

    def forward(self, x, cond=None):
        t = self.inn(x, cond)
        # if self.nullspace_split:
        #     _, _, V = torch.svd(list(self.layers_A.children())[0].weight, some=False)
        #     A = t_ @ V[: self.latent_dim + 1].T
        #     t = t_ @ V[self.latent_dim + 1 :].T
        # else:
        A = self.layers_A(t)
        if not self.soft_constraint:
            exp_A = torch.exp(A - A.max())
            A = exp_A / torch.sum(
                exp_A ** self.weight_norm_exp, dim=1, keepdim=True
            ) ** (1 / self.weight_norm_exp)

            if self.aa_allow_negative:
                A = (
                    self.aa_weights_max - self.aa_weights_min
                ) * A + self.aa_weights_min
                A = A / torch.sum(A, dim=1, keepdim=True)

            A = self.weight_norm_constraint * A

        if self.aa_weights_noise:
            A = A + self.aa_weights_noise * torch.randn_like(A)

        B = self.layers_B(t)
        if not self.soft_constraint:
            exp_B = torch.exp(B - B.max())
            B = exp_B / torch.sum(
                exp_B ** self.weight_norm_exp, dim=1, keepdim=True
            ) ** (1 / self.weight_norm_exp)

            if self.aa_allow_negative:
                B = (
                    self.aa_weights_max - self.aa_weights_min
                ) * B + self.aa_weights_min
                B = B / torch.sum(B, dim=1, keepdim=True)

            B = self.weight_norm_constraint * B

        if self.aa_weights_noise:
            B = B + self.aa_weights_noise * torch.randn_like(B)

        return t, A, B.T

    def sample(self, A, cond=None, t=None):
        # if self.nullspace_split:
        #     _, _, V = torch.svd(list(self.layers_A.children())[0].weight, some=False)

        if self.interpolation == "linear":
            Az = (
                torch.einsum(
                    "bj, bjk -> bk", A, self.z_arch[self.condition2labels(cond)]
                )
                if self.z_per_class
                else torch.einsum("bj, jk -> bk", A, self.z_arch)
            )
        elif self.interpolation == "slerp":
            A_ = torch.sin(A * np.pi * 2 / 3)
            Az = (
                torch.einsum(
                    "bj, bjk -> bk", A_, self.z_arch[self.condition2labels(cond)]
                )
                / np.sin(np.pi * 2 / 3)
                if self.z_per_class
                else torch.einsum("bj, jk -> bk", A_, self.z_arch)
                / np.sin(np.pi * 2 / 3)
            )
        t = self.layers_C(Az)
        if self.nullspace_split:
            # print(f"V: {V.shape}")
            # print(f"t: {t.shape}")
            # print(f"Az: {Az.shape}")
            mA = list(self.layers_A.children())[0].weight
            imA = torch.pinverse(mA)
            t = t + ((torch.eye(t.shape[1], device=device) - (imA @ mA)) @ t.T).T
        sideinfo = self.layers_sideinfo(Az)
        return self.inn.sample(t, cond), sideinfo

    def compute_losses(self, samples, labels):
        losses = {}
        cond = self.labels2condition(labels)
        t, A, B = self(samples, cond)

        # print(f"t: {t}")
        # print(f"A: {A}")
        # print(f"B: {B}")

        recreated, sideinfo = self.sample(A, cond)

        sample_latent_mean = (
            torch.einsum("bj, bjk -> bk", A, self.z_arch[self.condition2labels(cond)])
            if self.z_per_class
            else torch.einsum("bj, jk -> bk", A, self.z_arch)
        )
        at_loss = torch.mean(torch.norm(B @ sample_latent_mean - self.z_arch, dim=1))
        recon_loss = torch.norm(recreated - samples)
        class_loss = nn.functional.cross_entropy(sideinfo, labels)
        jac_loss = -torch.mean(self.inn.inn.log_jacobian(run_forward=False))

        # if not self.fix_inn:
        neg_log_likeli = self.inn.negative_log_likelihood(t, labels)
        # losses["Jac"] = self.lambda_jac * jac_loss
        losses["NLL"] = self.lambda_nll * neg_log_likeli
        losses["AT"] = self.lambda_at * at_loss
        losses["Recon"] = self.lambda_recon * recon_loss
        losses["Class"] = self.lambda_class * class_loss
        if self.use_proto_z:
            proto_loss = torch.sum(
                (self.z_arch[-1] - torch.mean(sample_latent_mean, dim=0)) ** 2
            )
            losses["Proto"] = self.lambda_proto * proto_loss
        if self.soft_constraint:
            losses["Constraint"] = self.lambda_constraint * (
                torch.sum((1 - torch.sum(A, dim=1)) ** 2) + torch.sum(A < 0)
            )

        return losses

    def labels2condition(self, labels):
        return self.inn.labels2condition(labels)

    def condition2labels(self, cond):
        return torch.where(cond[-1])[-1]

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(
            dict(filter(lambda x: "tmp" not in x[0], torch.load(path).items()))
        )


class IB_INN(nn.Module):
    def __init__(self, path):
        super().__init__()

        args = configparser.ConfigParser()
        args.read("~/Projects/IB-INN/default.ini")

        args.read(path / "conf.ini")

        output_base_dir = args["checkpoints"]["global_output_folder"]
        output_dir = os.path.join(output_base_dir, args["checkpoints"]["base_name"])
        args["checkpoints"]["output_dir"] = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.args = args

        init_latent_scale = eval(self.args["model"]["mu_init"])
        weight_init = eval(self.args["model"]["weight_init"])
        self.dataset = self.args["data"]["dataset"]
        self.ch_pad = eval(self.args["data"]["pad_noise_channels"])
        self.feed_forward = eval(self.args["ablations"]["feed_forward_resnet"])
        self.feed_forward_revnet = eval(self.args["ablations"]["feed_forward_irevnet"])
        print(self.dataset)

        if self.dataset == "MNIST":
            self.dims = (32, 32)
            self.input_channels = 1
            self.ndim_tot = int(np.prod(self.dims))
            self.n_classes = 10
        elif self.dataset in ["CIFAR10", "CIFAR100"]:
            self.dims = (3 + self.ch_pad, 32, 32)
            self.input_channels = 3 + self.ch_pad
            self.ndim_tot = int(np.prod(self.dims))
            if self.dataset == "CIFAR10":
                self.n_classes = 10
            else:
                self.n_classes = 100
        else:
            raise ValueError(f"what is this dataset, {args['data']['dataset']}?")

        # if self.feed_forward_revnet:
        #     self.feed_forward = True
        #     self.inn = inn_architecture.construct_irevnet(self)
        # elif self.feed_forward:
        #     self.inn = feed_forward_architecture.constuct_resnet(self.args)
        # else:
        self.inn = inn_architecture.constuct_inn(self)

        mu_populate_dims = self.ndim_tot
        init_scale = init_latent_scale / np.sqrt(2 * mu_populate_dims // self.n_classes)
        self.mu = nn.Parameter(torch.zeros(1, self.n_classes, self.ndim_tot))
        self.mu_empirical = eval(self.args["training"]["empirical_mu"])

        for k in range(mu_populate_dims // self.n_classes):
            self.mu.data[
                0, :, self.n_classes * k : self.n_classes * (k + 1)
            ] = init_scale * torch.eye(self.n_classes)

        self.phi = nn.Parameter(torch.zeros(self.n_classes))

        self.trainable_params = list(self.inn.parameters())
        self.trainable_params = list(
            filter(lambda p: p.requires_grad, self.trainable_params)
        )

        self.train_mu = eval(self.args["training"]["train_mu"])
        self.train_phi = eval(self.args["training"]["train_mu"])
        self.train_inn = True

        optimizer = self.args["training"]["optimizer"]

        for p in self.trainable_params:
            p.data *= weight_init

        self.trainable_params += [self.mu, self.phi]
        base_lr = float(self.args["training"]["lr"])

        optimizer_params = [
            {"params": list(filter(lambda p: p.requires_grad, self.inn.parameters()))},
        ]

        if self.train_mu:
            optimizer_params.append(
                {
                    "params": [self.mu],
                    "lr": base_lr * float(self.args["training"]["lr_mu"]),
                    "weight_decay": 0.0,
                }
            )
            if optimizer == "SGD":
                optimizer_params[-1]["momentum"] = float(
                    self.args["training"]["sgd_momentum_mu"]
                )
            if optimizer == "ADAM":
                optimizer_params[-1]["betas"] = eval(
                    self.args["training"]["adam_betas_mu"]
                )
            if optimizer == "AGGMO":
                optimizer_params[-1]["betas"] = eval(
                    self.args["training"]["aggmo_betas_mu"]
                )

        if self.train_phi:
            optimizer_params.append(
                {
                    "params": [self.phi],
                    "lr": base_lr * float(self.args["training"]["lr_phi"]),
                    "weight_decay": 0.0,
                }
            )
            if optimizer == "SGD":
                optimizer_params[-1]["momentum"] = float(
                    self.args["training"]["sgd_momentum_phi"]
                )
            if optimizer == "ADAM":
                optimizer_params[-1]["betas"] = eval(
                    self.args["training"]["adam_betas_phi"]
                )
            if optimizer == "AGGMO":
                optimizer_params[-1]["betas"] = eval(
                    self.args["training"]["aggmo_betas_phi"]
                )

        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                optimizer_params,
                base_lr,
                momentum=float(self.args["training"]["sgd_momentum"]),
                weight_decay=float(self.args["training"]["weight_decay"]),
            )
        elif optimizer == "ADAM":
            self.optimizer = torch.optim.Adam(
                optimizer_params,
                base_lr,
                betas=eval(self.args["training"]["adam_betas"]),
                weight_decay=float(self.args["training"]["weight_decay"]),
            )
        elif optimizer == "AGGMO":
            import aggmo

            self.optimizer = aggmo.AggMo(
                optimizer_params,
                base_lr,
                betas=eval(self.args["training"]["aggmo_betas"]),
                weight_decay=float(self.args["training"]["weight_decay"]),
            )
        else:
            raise ValueError(f"what is this optimizer, {optimizer}?")

    def cluster_distances(self, z, y=None):

        if y is not None:
            mu = torch.mm(z.t().detach(), y.round())
            mu = mu / torch.sum(y, dim=0, keepdim=True)
            mu = mu.t().view(1, self.n_classes, -1)
            mu = 0.005 * mu + 0.995 * self.mu.data
            self.mu.data = mu.data

        z_i_z_i = torch.sum(z ** 2, dim=1, keepdim=True)  # batchsize x n_classes
        mu_j_mu_j = torch.sum(self.mu ** 2, dim=2)  # 1 x n_classes
        z_i_mu_j = torch.mm(z, self.mu.squeeze().t())  # batchsize x n_classes

        return -2 * z_i_mu_j + z_i_z_i + mu_j_mu_j

    def mu_pairwise_dist(self):

        mu_i_mu_j = self.mu.squeeze().mm(self.mu.squeeze().t())
        mu_i_mu_i = torch.sum(self.mu.squeeze() ** 2, 1, keepdim=True).expand(
            self.n_classes, self.n_classes
        )

        dist = mu_i_mu_i + mu_i_mu_i.t() - 2 * mu_i_mu_j
        return torch.masked_select(
            dist, (1 - torch.eye(self.n_classes).cuda()).bool()
        ).clamp(min=0.0)

    def labels2condition(self, labels):
        """Convert a tensor of class labels to conditions for this network"""
        cond_tensor = torch.zeros(labels.size(0), 10, device=device)
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.0)

        fill = torch.zeros((10, 10, 32, 32), device=device)
        for i in range(10):
            fill[i, i, :, :] = 1

        cond = [
            fill[:, :, : 32 // 2, : 32 // 2][labels],
            fill[:, :, : 32 // 4, : 32 // 4][labels],
            cond_tensor,
        ]
        return cond

    def forward(self, x, y=None, loss_mean=True):

        # if self.feed_forward:
        #     return self.losses_feed_forward(x, y, loss_mean)

        z = self.inn(x)
        # jac = self.inn.log_jacobian(run_forward=False)

        # log_wy = torch.log_softmax(self.phi, dim=0).view(1, -1)

        # if self.mu_empirical and y is not None and self.inn.training:
        #     zz = self.cluster_distances(z, y)
        # else:
        #     zz = self.cluster_distances(z)

        # losses = {'L_x_tr':    (- torch.logsumexp(- 0.5 * zz + log_wy, dim=1) - jac ) / self.ndim_tot,
        #           'logits_tr': - 0.5 * zz}

        # log_wy = log_wy.detach()
        # if y is not None:
        #     losses['L_cNLL_tr'] = (0.5 * torch.sum(zz * y.round(), dim=1) - jac) / self.ndim_tot
        #     losses['L_y_tr'] = torch.sum((torch.log_softmax(- 0.5 * zz + log_wy, dim=1) - log_wy) * y, dim=1)
        #     losses['acc_tr'] = torch.mean((torch.max(y, dim=1)[1]
        #                                 == torch.max(losses['logits_tr'].detach(), dim=1)[1]).float())

        # if loss_mean:
        #     for k,v in losses.items():
        #         losses[k] = torch.mean(v)

        return z

    def losses_feed_forward(self, x, y=None, loss_mean=True):
        logits = self.inn(x)

        losses = {"logits_tr": logits, "L_x_tr": torch.zeros_like(logits[:, 0])}

        if y is not None:
            ly = torch.sum(torch.log_softmax(logits, dim=1) * y, dim=1)
            acc = torch.mean(
                (torch.max(y, dim=1)[1] == torch.max(logits.detach(), dim=1)[1]).float()
            )
            losses["L_y_tr"] = ly
            losses["acc_tr"] = acc
            losses["L_cNLL_tr"] = torch.zeros_like(ly)

        if loss_mean:
            for k, v in losses.items():
                losses[k] = torch.mean(v)

        return losses

    def validate(self, x, y, eval_mode=True):
        is_train = self.inn.training
        if eval_mode:
            self.inn.eval()

        with torch.no_grad():
            losses = self.forward(x, y, loss_mean=False)
            l_x, class_nll, l_y, logits, acc = (
                losses["L_x_tr"].mean(),
                losses["L_cNLL_tr"].mean(),
                losses["L_y_tr"].mean(),
                losses["logits_tr"],
                losses["acc_tr"],
            )

            mu_dist = torch.mean(torch.sqrt(self.mu_pairwise_dist()))

        if is_train:
            self.inn.train()

        return {
            "L_x_val": l_x,
            "L_cNLL_val": class_nll,
            "logits_val": logits,
            "L_y_val": l_y,
            "acc_val": acc,
            "delta_mu_val": mu_dist,
        }

    def reset_mu(self, dataset):
        mu = torch.zeros(1, self.n_classes, self.ndim_tot).cuda()
        counter = 0

        with torch.no_grad():
            for x, l in dataset.train_loader:
                x, y = x.cuda(), dataset.onehot(l.cuda(), 0.05)
                z = self.inn(x)
                mu_batch = torch.mm(z.t().detach(), y.round())
                mu_batch = mu_batch / torch.sum(y, dim=0, keepdim=True)
                mu_batch = mu_batch.t().view(1, self.n_classes, -1)

                mu += mu_batch
                counter += 1

            mu /= counter
        self.mu.data = mu.data

    def sample(self, y, temperature=1.0):
        # z = temperature * torch.randn(y.shape[0], self.ndim_tot).cuda()
        # mu = torch.sum(y.round().view(-1, self.n_classes, 1) * self.mu, dim=1)
        return self.inn(y, rev=True)

    def save(self, fname):
        torch.save(
            {
                "inn": self.inn.state_dict(),
                "mu": self.mu,
                "phi": self.phi,
                "opt": self.optimizer.state_dict(),
            },
            fname,
        )

    def load(self, fname):
        data = torch.load(fname)
        data["inn"] = {k: v for k, v in data["inn"].items() if "tmp_var" not in k}
        self.inn.load_state_dict(data["inn"])
        self.mu.data.copy_(data["mu"].data)
        self.phi.data.copy_(data["phi"].data)
        try:
            pass
        except:
            print("loading the optimizer went wrong, skipping")


def get_model(name):
    if name == "INN":
        return INN
    elif name == "INN_AA":
        return INN_AA
    elif name == "DCGAN":
        raise NotImplementedError()
    else:
        raise Exception(f"Model type {name} is not known")

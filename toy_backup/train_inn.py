import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.utils.data as data

import FrEIA.framework as Ff
import FrEIA.modules as Fm

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 512
LR = 0.001
N_EPOCHS = 300
SAMPLE_INTERVAL = 200


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(), nn.Linear(512, c_out))


def MMD_multiscale(x, y):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    for a in [0.05, 0.2, 0.9]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def generate_toy_data(N=100000, uniform_range=50):
    mu = 3 * np.array([[1, 1], [-1, -1]])
    # mu = 3*np.array([
    #     (1, 0),
    #     (-1, 0),
    #     (0, 1),
    #     (0, -1),
    #     (1. / np.sqrt(2), 1. / np.sqrt(2)),
    #     (1. / np.sqrt(2), -1. / np.sqrt(2)),
    #     (-1. / np.sqrt(2), 1. / np.sqrt(2)),
    #     (-1. / np.sqrt(2), -1. / np.sqrt(2))
    # ])
    samples = np.concatenate(
        [np.random.multivariate_normal(m, 1 * np.eye(2), N) for m in mu]
    )
    # samples = np.concatenate([
    #     samples,
    #     np.random.uniform(-uniform_range, uniform_range, (N//3, 2))
    # ])

    labels = np.concatenate(
        [
            np.zeros(N),
            np.ones(N),
            # 2*np.ones(N),
            # 3*np.ones(N),
            # 4*np.ones(N),
            # 5*np.ones(N),
            # 6*np.ones(N),
            # 7*np.ones(N),
            # np.random.choice(np.arange(2), N//3)
        ]
    )

    # index = np.random.permutation(samples.shape[0])

    return torch.tensor(samples), torch.tensor(labels)


nodes = [Ff.InputNode(2, name="input")]

# Use a loop to produce a chain of coupling blocks
for k in range(2):
    nodes.append(
        Ff.Node(
            nodes[-1],
            Fm.GLOWCouplingBlock,
            {"subnet_constructor": subnet_fc, "clamp": 2.0},
            name=f"coupling_{k}",
        )
    )

nodes.append(Ff.OutputNode(nodes[-1], name="output"))
inn = Ff.ReversibleGraphNet(nodes)
inn.to(device)

optimizer = torch.optim.Adam(inn.parameters(), lr=LR, betas=(0.5, 0.9))

dataset = data.TensorDataset(*generate_toy_data())
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

xx, yy = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))

grid = (
    torch.stack([torch.tensor(xx.ravel()), torch.tensor(yy.ravel())])
    .to(device)
    .float()
    .t()
)

batches_done = 0
for epoch in range(N_EPOCHS):

    for i, (x, y) in enumerate(dataloader):

        if x.shape[0] != BATCH_SIZE:
            continue

        optimizer.zero_grad()

        # Configure input
        real_data = x.float().to(device=device)
        targets = y.to(device).long()

        output = inn(real_data)
        zz = torch.sum(output ** 2, dim=1)
        jac = inn.jacobian(run_forward=False)

        neg_log_likeli = 0.5 * zz - jac

        l = torch.mean(neg_log_likeli)
        l.backward(retain_graph=True)

        z = torch.randn(BATCH_SIZE, 2, device=device)

        x_rec = inn(output.data, rev=True)
        l_rev = torch.mean((real_data - x_rec) ** 2)

        x_rand = inn(z, rev=True)
        l_rev += MMD_multiscale(x_rand, real_data)
        l_rev.backward()

        optimizer.step()

        if batches_done % SAMPLE_INTERVAL == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d]"
                % (epoch, N_EPOCHS, batches_done % len(dataloader), len(dataloader))
            )

        batches_done += 1

    plt.figure()
    x, y = generate_toy_data(N=1000)
    z = torch.randn(BATCH_SIZE, 2, device=device)
    plt.scatter(x[:, 0], x[:, 1], alpha=0.3, marker=".")
    gen_data = inn(z, rev=True)
    plt.scatter(
        gen_data[:, 0].detach().cpu(), gen_data[:, 1].detach().cpu(), marker="."
    )
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"inn/generator_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

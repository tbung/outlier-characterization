import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


device = "cuda" if torch.cuda.is_available() else "cpu"

DIM = 512
BATCH_SIZE = 400
LR = 0.002
N_EPOCHS = 300
N_CRITIC = 5
SAMPLE_INTERVAL = 200
CLIP_VALUE = 0.01
LOSS = "gp"
BETA = 10


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


def generate_image(true_dist, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 512
    RANGE = 10

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype="float32")
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))

    points_v = torch.tensor(points, requires_grad=True, device=device)
    disc_map = netD(points_v)

    plt.figure()
    plt.xlim((-RANGE, RANGE))
    plt.ylim((-RANGE, RANGE))

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(
        x, y, disc_map.detach().cpu().numpy().reshape((len(x), len(y))).transpose()
    )

    plt.scatter(
        true_dist[:, 0].cpu(), true_dist[:, 1].cpu(), c="orange", marker="+", alpha=0.3
    )

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.savefig(f"pls/discriminator_{epoch:08d}.png", figsize=(8, 8))
    plt.close()


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1, device=device, requires_grad=True)
    alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _Generator(nn.Module):
    def __init__(self):
        super(_Generator, self).__init__()

        main = nn.Sequential(
            nn.Linear(100, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 2),
        )
        self.main = main

    def forward(self, noise):
        output = self.main(noise)
        return output


class _Discriminator(nn.Module):
    def __init__(self):
        super(_Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
            nn.Sigmoid()
            # nn.BatchNorm1d(1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs).view(-1)
        # return torch.tanh(output)
        return output


def Generator(n_gpu, nz, ngf, nc):
    model = _Generator()
    model.apply(weights_init)
    return model


def Discriminator(n_gpu, nc, ndf):
    model = _Discriminator()
    model.apply(weights_init)
    return model


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(True),
            # nn.BatchNorm1d(100),
            nn.Linear(500, 500),
            nn.ReLU(True),
            # nn.Linear(512, 1024),
            # nn.ReLU(True),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, 2046),
            # nn.ReLU(True),
            # nn.BatchNorm1d(2046),
            # nn.Linear(2046, 2046),
            # nn.ReLU(True),
            # nn.BatchNorm1d(2046),
            nn.Linear(500, 2),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def train_classifier():
    model = Classifier().to(device)
    netG = Generator(1, nz, 64, 3).to(device)  # ngpu, nz, ngf, nc
    netD = Discriminator(1, 3, 64).to(device)  # ngpu, nc, ndf

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))

    dataset = data.TensorDataset(*generate_toy_data(N=10000))
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    testset = data.TensorDataset(*generate_toy_data(N=1000))
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    ood_samples = np.random.uniform(-10, 10, [10000, 2])
    i = (
        stats.multivariate_normal.pdf(ood_samples, [1, 1], 1 * np.eye(2))
        + stats.multivariate_normal.pdf(ood_samples, [-1, -1], 1 * np.eye(2))
    ) / 2

    i /= np.max(i)
    ood_samples = ood_samples[np.random.binomial(1, i) != 1]

    xx, yy = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
    grid = (
        torch.stack([torch.tensor(xx.ravel()), torch.tensor(yy.ravel())])
        .to(device)
        .float()
        .t()
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        # scheduler.step()
        total_loss = 0
        for i, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            inputs = x.to(device).float()
            targets = y.to(device).long()

            samples = torch.tensor(
                ood_samples[np.random.permutation(ood_samples.shape[0])][
                    : inputs.size(0)
                ],
                device=device,
            ).float()

            uniform_dist = torch.Tensor(inputs.size(0), 2).fill_((1 / 2)).to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets) + 2 * torch.nn.functional.kl_div(
                F.log_softmax(model(samples)), uniform_dist
            )
            total_loss += loss

            loss.backward()
            optimizer.step()

            # optimizerD.zero_grad()
            # output = netD(inputs)
            # errD_real = F.binary_cross_entropy(output, torch.ones_like(output, device=device))
            # errD_real.backward()

            # # train with fake
            # output = netD(samples)
            # errD_fake = F.binary_cross_entropy(output, torch.zeros_like(output, device=device))
            # errD_fake.backward()
            # optimizerD.step()

            ############################
            ## (2) Update G network    #
            ############################
            # optimizerG.zero_grad()
            # # Original GAN loss
            # fake = netG(torch.randn(inputs.size(0), 100, device=device))
            # output = netD(fake)
            # # errG = F.binary_cross_entropy(output, torch.ones_like(output, device=device))

            # # minimize the true distribution
            # KL_fake_output = F.log_softmax(model(fake))
            # errG_KL = F.kl_div(KL_fake_output, uniform_dist)*2
            # generator_loss = BETA*errG_KL
            # generator_loss.backward()
            # optimizerG.step()
        print(f"Loss: {loss/len(dataloader)}")

        accuracy = 0
        for i, (x, y) in enumerate(testloader):
            inputs = x.to(device).float()
            targets = y.to(device).long()

            outputs = torch.argmax(torch.nn.functional.softmax(model(inputs)), dim=1)
            # print(f"Targets: {targets}")
            # print(f"Outputs: {outputs}")
            accuracy += (outputs == targets).float().mean()

        if epoch % 10 == 0:
            zz = torch.cat(
                [model(grid[i : i + 100]) for i in range(0, grid.shape[0], 100)]
            )

            plt.contourf(
                xx,
                yy,
                np.argmax(zz.detach().cpu(), axis=1).reshape(1000, 1000),
                cmap="tab10",
                vmax=9,
                alpha=0.5,
            )
            x, y = generate_toy_data(N=1000)
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9, marker=".")

            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.savefig(f"pls/classifier_{epoch:03d}.png", figsize=(8, 8))
            plt.close()

            plt.contourf(
                xx,
                yy,
                np.abs(np.diff(zz.detach().cpu(), axis=1)).reshape(1000, 1000),
                cmap="inferno",
            )
            x, y = generate_toy_data(N=1000)
            plt.scatter(
                x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9, alpha=0.3, marker="."
            )

            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.savefig(f"pls/confidence_{epoch:03d}.png", figsize=(8, 8))
            plt.close()

            del zz

        print(f"Test Accuracy: {accuracy/len(testloader)}")

    return model, netG, netD


nz = 100
netGin = Generator(1, nz, 64, 3).to(device)  # ngpu, nz, ngf, nc
# netD = Discriminator(1, 3, 64).to(device)  # ngpu, nc, ndf
model, netG, netD = train_classifier()
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss().to(device)
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1).to(device)

print("Setup optimizer")
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.999))
optimizerGin = torch.optim.Adam(netGin.parameters(), lr=LR, betas=(0.5, 0.999))
decreasing_lr = 60

dataset = data.TensorDataset(*generate_toy_data())
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def train(epoch):
    model.train()
    for batch_idx, (d, target) in enumerate(dataloader):

        uniform_dist = torch.Tensor(d.size(0), 2).fill_((1 / 2)).to(device)
        d = d.to(device).float()
        target = target.to(device).long()

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        optimizerD.zero_grad()
        output = netD(d)
        errD_real = criterion(output, torch.ones_like(output, device=device))
        errD_real.backward()

        # train with fake
        noise = torch.FloatTensor(d.size(0), nz).normal_(0, 1).to(device)
        fake = netG(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, torch.zeros_like(output, device=device))
        errD_fake.backward()

        # train with fake
        noise = torch.FloatTensor(d.size(0), nz).normal_(0, 1).to(device)
        fake = netGin(noise)
        output = netD(fake.detach())
        errD_fake = criterion(output, torch.zeros_like(output, device=device))
        errD_fake.backward()
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        fake = netG(noise)
        output = netD(fake)
        errG = criterion(output, torch.ones_like(output, device=device))

        # minimize the true distribution
        KL_fake_output = F.log_softmax(model(fake))
        errG_KL = F.kl_div(KL_fake_output, uniform_dist) * 2
        generator_loss = errG + BETA * errG_KL
        generator_loss.backward()
        optimizerG.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerGin.zero_grad()
        # Original GAN loss
        fake = netGin(noise)
        output = netD(fake)
        errG = criterion(output, torch.ones_like(output, device=device))
        errG.backward()
        optimizerGin.step()

        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        # optimizer.zero_grad()
        # output = F.log_softmax(model(d))
        # loss = F.nll_loss(output, target)

        # # KL divergence
        # noise = torch.FloatTensor(d.size(0), nz).normal_(0, 1).to(device)
        # fake = netG(noise)
        # KL_fake_output = F.log_softmax(model(fake))
        # KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*2
        # total_loss = loss + BETA*KL_loss_fake
        # total_loss.backward()
        # optimizer.step()

        if batch_idx % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d]"
                % (epoch, N_EPOCHS, batch_idx, len(dataloader))
            )

    plt.figure()
    plt.scatter(d[:, 0].detach().cpu(), d[:, 1].detach().cpu(), alpha=0.3, marker=".")
    fake = netG(noise)
    plt.scatter(fake[:, 0].detach().cpu(), fake[:, 1].detach().cpu(), marker=".")
    fake = netGin(noise)
    plt.scatter(fake[:, 0].detach().cpu(), fake[:, 1].detach().cpu(), marker=".")
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"pls/generator_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

    generate_image(d, epoch)

    xx, yy = np.meshgrid(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
    grid = (
        torch.stack([torch.tensor(xx.ravel()), torch.tensor(yy.ravel())])
        .to(device)
        .float()
        .t()
    )
    zz = torch.cat([model(grid[i : i + 100]) for i in range(0, grid.shape[0], 100)])

    plt.contourf(
        xx,
        yy,
        np.argmax(zz.detach().cpu(), axis=1).reshape(1000, 1000),
        cmap="tab10",
        vmax=9,
        alpha=0.5,
    )
    x, y = generate_toy_data(N=1000)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9, marker=".")

    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"pls/classifier_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

    plt.contourf(
        xx,
        yy,
        np.abs(np.diff(zz.detach().cpu(), axis=1)).reshape(1000, 1000),
        cmap="inferno",
        alpha=1,
    )
    x, y = generate_toy_data(N=1000)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9, alpha=0.3, marker=".")

    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"pls/confidence_{epoch:08d}.png", figsize=(8, 8))
    plt.close()


for epoch in range(1, N_EPOCHS + 1):
    train(epoch)
    if epoch % decreasing_lr == (decreasing_lr - 1):
        optimizerG.param_groups[0]["lr"] *= 0.1
        optimizerGin.param_groups[0]["lr"] *= 0.1
        optimizerD.param_groups[0]["lr"] *= 0.1
        optimizer.param_groups[0]["lr"] *= 0.1
    # if epoch % 20 == 0:
    #     # do checkpointing
    #     torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
    #     torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
    #     torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))

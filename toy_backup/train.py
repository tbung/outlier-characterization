import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.utils.data as data


device = "cuda" if torch.cuda.is_available() else "cpu"

DIM = 512
BATCH_SIZE = 256
LR = 1e-4
LAMBDA = 0.001
N_EPOCHS = 300
N_CRITIC = 5
SAMPLE_INTERVAL = 200
LOSS = "gp"
BETA = 50
ALPHA = 1


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


def generate_image(true_dist):
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
    disc_map = discriminator(points_v)

    plt.figure()
    plt.xlim((-RANGE, RANGE))
    plt.ylim((-RANGE, RANGE))

    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    plt.contour(
        x, y, disc_map.detach().cpu().numpy().reshape((len(x), len(y))).transpose()
    )

    plt.scatter(true_dist[:, 0].cpu(), true_dist[:, 1].cpu(), c="orange", marker="+")

    ax = plt.gca()
    ax.set_aspect("equal")

    plt.savefig(f"discriminator_{batches_done:08d}.png", figsize=(8, 8))
    plt.close()

    # plt.clf()
    # plt.pcolor(x, y, disc_map.detach().cpu().numpy().reshape((len(x), len(y))).transpose())
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # plt.savefig(FIGURE_PATH / f'cmap{frame_index[0]:03d}.png', figsize=(8,8))


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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(2, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            # nn.BatchNorm1d(DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
            # nn.BatchNorm1d(1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs).view(-1)
        # return torch.tanh(output)
        return output


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

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4
    )

    dataset = data.TensorDataset(*generate_toy_data(N=10000))
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    testset = data.TensorDataset(*generate_toy_data(N=1000))
    testloader = data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    ood_samples = np.random.uniform(-10, 10, [10000, 2])
    i = (
        stats.multivariate_normal.pdf(ood_samples, [1, 1], np.eye(2))
        + stats.multivariate_normal.pdf(ood_samples, [-1, -1], np.eye(2))
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

    for epoch in range(100):
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
            loss = nn.functional.cross_entropy(
                outputs, targets
            ) + 2 * torch.nn.functional.kl_div(
                nn.functional.log_softmax(model(samples)), uniform_dist
            )
            total_loss += loss

            loss.backward()
            optimizer.step()

        print(f"Loss: {loss/len(dataloader)}")

        accuracy = 0
        for i, (x, y) in enumerate(testloader):
            inputs = x.to(device).float()
            targets = y.to(device).long()

            outputs = torch.argmax(torch.nn.functional.softmax(model(inputs)), dim=1)
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
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9)

            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.savefig(f"classifier_{epoch:03d}.png", figsize=(8, 8))
            plt.close()

            plt.contourf(
                xx,
                yy,
                np.abs(np.diff(zz.detach().cpu(), axis=1)).reshape(1000, 1000),
                cmap="inferno",
            )
            x, y = generate_toy_data(N=1000)
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9)

            plt.xlim((-10, 10))
            plt.ylim((-10, 10))
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.savefig(f"confidence_{epoch:03d}.png", figsize=(8, 8))

            del zz

        print(f"Test Accuracy: {accuracy/len(testloader)}")

    return model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


generatorOut = Generator().to(device=device)
generatorIn = Generator().to(device=device)
discriminator = Discriminator().to(device=device)
generatorIn.apply(weights_init)
generatorOut.apply(weights_init)
discriminator.apply(weights_init)

classifier = train_classifier()

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_Gin = torch.optim.Adam(generatorIn.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_Gout = torch.optim.Adam(generatorOut.parameters(), lr=LR, betas=(0.5, 0.9))
optimizer_C = torch.optim.Adam(
    classifier.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-3
)

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

        # Configure input
        real_data = x.float().to(device=device)
        targets = y.to(device).long()
        uniform_dist = torch.Tensor(real_data.size(0), 2).fill_((1 / 2)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        # z = torch.tensor(np.random.normal(0, 1, (BATCH_SIZE, 2)),
        #                  device=device).float()
        z = torch.randn(BATCH_SIZE, 100, device=device)

        # Generate a batch of images
        fake_data_in = generatorIn(z).detach()
        fake_data_out = generatorOut(z).detach()
        # Adversarial loss
        D_real = discriminator(real_data)
        D_fake = discriminator(fake_data_in) + ALPHA * discriminator(fake_data_out)

        loss_D = -(torch.mean(D_real) - torch.mean(D_fake))

        loss_D.backward(retain_graph=True)

        if LOSS == "gp":
            if real_data.size(0) != BATCH_SIZE:
                continue
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(
                discriminator, real_data.data, fake_data_in.data
            )
            gradient_penalty += ALPHA * calc_gradient_penalty(
                discriminator, real_data.data, fake_data_out.data
            )
            gradient_penalty.backward()
        elif LOSS == "var":
            penalty = LAMBDA * torch.exp(
                torch.sqrt(
                    torch.log(D_real.var(0)) ** 2 + torch.log(D_fake.var(0)) ** 2
                )
            )
            penalty.backward()

        optimizer_D.step()

        # Train the generator every n_critic iterations
        if i % N_CRITIC == 0:

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_Gout.zero_grad()

            # Generate a batch of images
            gen_data = generatorOut(z)
            KL_fake_output = nn.functional.log_softmax(classifier(gen_data))
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_data))
            loss_G += BETA * 2 * nn.functional.kl_div(KL_fake_output, uniform_dist)

            loss_G.backward()
            optimizer_Gout.step()

            optimizer_Gin.zero_grad()

            # Generate a batch of images
            gen_data = generatorIn(z)
            # Adversarial loss
            loss_G = -torch.mean(discriminator(gen_data))

            loss_G.backward()
            optimizer_Gin.step()

        if batches_done % SAMPLE_INTERVAL == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    N_EPOCHS,
                    batches_done % len(dataloader),
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                )
            )

        batches_done += 1

    plt.figure()
    x, y = generate_toy_data(N=1000)
    plt.scatter(x[:, 0], x[:, 1], alpha=0.3, marker=".")
    gen_data = generatorOut(z)
    plt.scatter(
        gen_data[:, 0].detach().cpu(), gen_data[:, 1].detach().cpu(), marker="."
    )
    gen_data = generatorIn(z)
    plt.scatter(
        gen_data[:, 0].detach().cpu(), gen_data[:, 1].detach().cpu(), marker="."
    )
    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"generator_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

    generate_image(real_data)

    zz = torch.cat(
        [classifier(grid[i : i + 100]) for i in range(0, grid.shape[0], 100)]
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
    plt.savefig(f"classifier_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

    plt.contourf(
        xx,
        yy,
        np.abs(np.diff(zz.detach().cpu(), axis=1)).reshape(1000, 1000),
        cmap="inferno",
    )
    x, y = generate_toy_data(N=1000)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10", vmax=9, marker=".")

    plt.xlim((-10, 10))
    plt.ylim((-10, 10))
    ax = plt.gca()
    ax.set_aspect("equal")
    plt.savefig(f"confidence_{epoch:08d}.png", figsize=(8, 8))
    plt.close()

    del zz

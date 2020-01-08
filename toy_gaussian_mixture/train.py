from pathlib import Path
from crayons import cyan, white, red
from tqdm import tqdm

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import config as c
from models import Generator, Discriminator, Classifier, INN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINTS_PATH = Path('./checkpoints')
CHECKPOINTS_PATH.mkdir(exist_ok=True)

writer = SummaryWriter()

config_dict = {}
for k in dir(c):
    if k.startswith('_'):
        continue
    v = eval(f'c.{k}')
    config_dict[k] = v

writer.add_hparams(config_dict, {})


def generate_toy_data(N=100000, uniform_range=50):
    mu = 3*np.array([[1, 1], [-1, -1]])

    samples = np.concatenate([
        np.random.multivariate_normal(m, 1*np.eye(2), N)
        for m in mu
    ])

    labels = np.concatenate([
        np.zeros(N),
        np.ones(N),
    ])

    return torch.tensor(samples), torch.tensor(labels)


def generate_ood_data(N):
    ood_samples = np.empty((0, 2))
    while ood_samples.shape[0] < N:
        samples = np.random.uniform(-10, 10, [N, 2])
        i = (stats.multivariate_normal.pdf(samples, [1, 1], 1*np.eye(2))
             + stats.multivariate_normal.pdf(samples, [-1, -1],
                                             1*np.eye(2)))/2

        i /= np.max(i)
        ood_samples = np.concatenate([
            ood_samples, samples[np.random.binomial(1, i) != 1]
        ])
    return torch.tensor(ood_samples[:N], device=device).float()


def plot_classifier(model, n_iter):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 1000),
                         np.linspace(-10, 10, 1000))

    grid = torch.stack([
        torch.tensor(xx.ravel()), torch.tensor(yy.ravel())
    ]).to(device).float().t()

    zz = torch.cat([
        model(grid[i:i+100]) for i in range(0, grid.shape[0], 100)
    ])

    x, y = generate_toy_data(N=1000)

    boundaries_fig, ax = plt.subplots()
    ax.contourf(xx, yy,
                np.argmax(zz.detach().cpu(), axis=1).reshape(1000, 1000),
                cmap='tab10', vmax=9, alpha=0.5)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10', vmax=9, marker='.')

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_aspect('equal')

    writer.add_figure('Classifier_Pretraining/Boundary', boundaries_fig,
                      n_iter, close=True)

    certainty_fig, ax = plt.subplots()
    ax.contourf(xx, yy,
                np.abs(np.diff(zz.detach().cpu(), axis=1)).reshape(1000, 1000),
                cmap='inferno')
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10', vmax=9, alpha=0.3,
               marker='.')

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_aspect('equal')

    writer.add_figure('Classifier_Pretraining/Certainty', certainty_fig,
                      n_iter, close=True)


def plot_generator(models, n_iter,):
    samples, _ = generate_toy_data(c.batch_size)
    fig, ax = plt.subplots()

    ax.scatter(samples[:, 0].detach().cpu(), samples[:, 1].detach().cpu(),
               alpha=0.3, marker='.')
    for model, rev in models:
        if rev:
            fake = model(noise, rev=True)
        else:
            fake = model(noise)
        ax.scatter(fake[:, 0].detach().cpu(), fake[:, 1].detach().cpu(),
                   marker='.')
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_aspect('equal')

    writer.add_figure('GAN_Training/Generator', fig, n_iter, close=True)


def plot_discriminator(model, n_iter):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 1000),
                         np.linspace(-10, 10, 1000))

    grid = torch.stack([
        torch.tensor(xx.ravel()), torch.tensor(yy.ravel())
    ]).to(device).float().t()

    zz = torch.cat([
        model(grid[i:i+100]) for i in range(0, grid.shape[0], 100)
    ])

    x, y = generate_toy_data(N=1000)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy,
                zz.detach().cpu().numpy().reshape(1000, 1000),
                cmap='inferno')
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap='tab10', vmax=9, marker='.')

    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
    ax.set_aspect('equal')

    writer.add_figure('GAN_Training/Discriminator', fig,
                      n_iter, close=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pretrain_classifier(restore=True):
    print(cyan("Pretraining Classifier", bold=True))

    if restore and (CHECKPOINTS_PATH / 'classifier_toy.pth').is_file():
        return torch.load(CHECKPOINTS_PATH / 'classifier_toy.pth')

    model = Classifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_classifier, weight_decay=1e-4)

    headers = ["Epoch", "Loss", "KL Div", "Accuracy"]
    loss_format = "{:>15}" * len(headers)

    print(white(loss_format.format(*headers), bold=True))

    for epoch in range(c.n_epochs_pretrain):

        loss_history = []

        for i, (x, y) in enumerate(tqdm(train_loader, leave=False)):
            losses = []

            optimizer.zero_grad()

            inputs = x.to(device).float()
            targets = y.to(device).long()
            samples = generate_ood_data(inputs.size(0))

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            losses.append(loss)

            uniform_dist = torch.Tensor(inputs.size(0), 2).fill_((1/2)).to(device)

            kl_loss = c.beta_classifier * 2 * F.kl_div(
                F.log_softmax(model(samples), dim=1),
                uniform_dist, reduction='batchmean'
            )
            kl_loss.backward()

            losses.append(kl_loss)

            optimizer.step()

            loss_history.append(losses)

        accuracy = 0
        for i, (x, y) in enumerate(test_loader):
            inputs = x.to(device).float()
            targets = y.to(device).long()

            outputs = torch.argmax(F.softmax(model(inputs), dim=1),
                                   dim=1)
            accuracy += (outputs == targets).float().mean()

        plot_classifier(model, epoch)

        print(loss_format.format(
            f"{epoch}",
            *[f"{l:.3f}" for l in np.mean(np.array(loss_history), axis=0)],
            f"{accuracy/len(test_loader):.1%}%"
        ))

    torch.save(model, CHECKPOINTS_PATH / 'classifier_toy.pth')

    return model


train_loader = data.DataLoader(data.TensorDataset(*generate_toy_data(N=10000)),
                               batch_size=c.batch_size, shuffle=True)

test_loader = data.DataLoader(data.TensorDataset(*generate_toy_data(N=1000)),
                              batch_size=c.batch_size, shuffle=False)

generators = []
if c.train_inner_gan:
    generator_in = Generator().to(device)
    generator_in.apply(weights_init)
    generators.append((generator_in, False))
    optimizer_Gin = torch.optim.Adam(generator_in.parameters(),
                                     lr=c.lr_generator,
                                     betas=[0.5, 0.999])
    scheduler_Gin = torch.optim.lr_scheduler.StepLR(
        optimizer_Gin, c.lr_schedule, c.lr_step
    )
elif c.train_inner_inn:
    generator_in = INN().to(device)
    generators.append((generator_in, True))
    optimizer_Gin = torch.optim.Adam(generator_in.parameters(),
                                     lr=c.lr_generator,
                                     betas=[0.5, 0.999])
    scheduler_Gin = torch.optim.lr_scheduler.StepLR(
        optimizer_Gin, c.lr_schedule, c.lr_step
    )

if c.train_outer_gan:
    generator_out = Generator().to(device)
    generator_out.apply(weights_init)
    generators.append((generator_out, False))
    optimizer_Gout = torch.optim.Adam(generator_out.parameters(),
                                      lr=c.lr_generator,
                                      betas=[0.5, 0.999])
    scheduler_Gout = torch.optim.lr_scheduler.StepLR(
        optimizer_Gout,  c.lr_schedule, c.lr_step
    )
    if c.pretrain_classifier:
        classifier = pretrain_classifier()
    else:
        classifier = Classifier().to(device)
elif c.train_outer_inn:
    generator_out = INN().to(device)
    generators.append((generator_out, True))
    optimizer_Gout = torch.optim.Adam(generator_out.parameters(),
                                      lr=c.lr_generator,
                                      betas=[0.5, 0.999])
    scheduler_Gout = torch.optim.lr_scheduler.StepLR(
        optimizer_Gout, c.lr_schedule, c.lr_step
    )
    if c.pretrain_classifier:
        classifier = pretrain_classifier()
    else:
        classifier = Classifier().to(device)

discriminator = Discriminator().to(device)
discriminator.apply(weights_init)
optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=c.lr_discriminator,
                               betas=[0.5, 0.999])
scheduler_D = torch.optim.lr_scheduler.StepLR(
    optimizer_D, c.lr_schedule, c.lr_step
)

print(cyan("Begin GAN Training", bold=True))
try:
    headers = ["Epoch", "errD_real"]
    if c.train_inner_gan:
        headers.extend(["errD_fake_in", "errG_in"])
    if c.train_inner_inn:
        headers.extend(["errD_fake_in", "nll_in", "rec_in"])
    if c.train_outer_gan:
        headers.extend(["errD_fake_out", "errG_out", "errG_kl"])
    if c.train_outer_inn:
        headers.extend(["errD_fake_out", "nll_out", "errG_kl"])
    if len(headers) == 7:
        headers.insert(3, headers.pop(4))
    if len(headers) == 8:
        headers.insert(3, headers.pop(5))

    loss_format = "{:>15}" * len(headers)
    print(white(loss_format.format(*headers), bold=True))

    for epoch in range(c.n_epochs):

        loss_history = []

        for n, (samples, targets) in enumerate(tqdm(train_loader, leave=False)):
            losses = []
            n_iter = n + (epoch * len(train_loader))

            uniform_dist = torch.Tensor(samples.size(0), 2).fill_((1/2)).to(device)
            samples = samples.to(device).float()
            targets = targets.to(device).long()

            ###########################
            # (1) Update D network    #
            ###########################
            # train with real
            optimizer_D.zero_grad()
            output = discriminator(samples)
            errD_real = F.binary_cross_entropy(
                output, torch.ones_like(output, device=device)
            )
            errD_real.backward()
            losses.append(errD_real)

            if c.train_inner_gan:
                noise = torch.randn((samples.size(0), c.nz), device=device)
                fake = generator_in(noise)
                output = discriminator(fake.detach())
                errD_fake = F.binary_cross_entropy(
                    output, torch.zeros_like(output, device=device)
                )
                errD_fake.backward()
                losses.append(errD_fake)

            if c.train_inner_inn:
                noise = torch.randn((samples.size(0), c.nz), device=device)
                fake = generator_in(noise, rev=True)
                output = discriminator(fake.detach())
                errD_fake = F.binary_cross_entropy(
                    output, torch.zeros_like(output, device=device)
                )
                errD_fake.backward()
                losses.append(errD_fake)

            if c.train_outer_gan:
                noise = torch.randn((samples.size(0), c.nz), device=device)
                fake = generator_out(noise)
                output = discriminator(fake.detach())
                errD_fake = F.binary_cross_entropy(
                    output, torch.zeros_like(output, device=device)
                )
                errD_fake.backward()
                losses.append(errD_fake)

            if c.train_outer_inn:
                noise = torch.randn((samples.size(0), c.nz), device=device)
                fake = generator_out(noise, rev=True)
                output = discriminator(fake.detach())
                errD_fake = F.binary_cross_entropy(
                    output, torch.zeros_like(output, device=device)
                )
                errD_fake.backward()
                losses.append(errD_fake)

            optimizer_D.step()

            writer.add_scalar('Loss/Discriminator', sum(losses), n_iter)

            ###########################
            # (2) Update G network    #
            ###########################
            if c.train_inner_gan:
                optimizer_Gin.zero_grad()

                noise = torch.randn((samples.size(0), c.nz), device=device)

                fake = generator_in(noise)
                output = discriminator(fake)
                errG = F.binary_cross_entropy(
                    output, torch.ones_like(output, device=device)
                )
                errG.backward()
                losses.append(errG)

                optimizer_Gin.step()

                writer.add_scalar('Loss/In-Dist Generator', errG, n_iter)

            if c.train_outer_gan:
                optimizer_Gout.zero_grad()

                noise = torch.randn((samples.size(0), c.nz), device=device)
                fake = generator_out(noise)
                output = discriminator(fake)
                errG = F.binary_cross_entropy(
                    output, torch.ones_like(output, device=device)
                )
                errG.backward(retain_graph=True)
                losses.append(errG.data)

                # minimize the true distribution
                kl_loss = c.beta_generator * 2 * F.kl_div(
                    F.log_softmax(classifier(fake), dim=1),
                    uniform_dist, reduction='batchmean'
                )
                kl_loss.backward()
                losses.append(kl_loss.data)

                optimizer_Gout.step()

                writer.add_scalar('Loss/Out-Dist Generator', errG, n_iter)
                writer.add_scalar('Loss/KL_Div', kl_loss, n_iter)

            if c.train_outer_inn:
                optimizer_Gout.zero_grad()

                output = generator_out(samples)
                zz = torch.sum(output**2, dim=1)
                jac = generator_out.jacobian(run_forward=False)

                neg_log_likeli = 0.5 * zz - jac

                errG = torch.mean(neg_log_likeli)
                errG.backward(retain_graph=True)
                losses.append(errG.data)

                fake = generator_out(noise, rev=True)

                # minimize the true distribution
                kl_loss = c.beta_generator * 2 * F.kl_div(
                    F.log_softmax(classifier(fake), dim=1),
                    uniform_dist, reduction='batchmean'
                )
                kl_loss.backward()

                losses.append(kl_loss.data)

                optimizer_Gout.step()

                writer.add_scalar('Loss/Negative Log-Likelihood Out-Dist', errG, n_iter)
                writer.add_scalar('Loss/Reconstruction Loss Out-Dist', kl_loss, n_iter)

            if c.train_inner_inn:
                optimizer_Gin.zero_grad()
                output = generator_in(samples)
                zz = torch.sum(output**2, dim=1)
                jac = generator_in.jacobian(run_forward=False)

                neg_log_likeli = 0.5 * zz - jac

                errG = torch.mean(neg_log_likeli)
                errG.backward(retain_graph=True)

                losses.append(errG.data)

                z = torch.randn(samples.size(0), 2, device=device)

                fake_samples = generator_in(z, rev=True)
                output = discriminator(fake_samples)
                l_rev = F.binary_cross_entropy(
                    output, torch.ones_like(output, device=device)
                )
                l_rev.backward()

                losses.append(l_rev.data)

                optimizer_Gin.step()

                writer.add_scalar('Loss/Negative Log-Likelihood In-Dist', errG, n_iter)
                writer.add_scalar('Loss/Reconstruction Loss In-Dist', l_rev, n_iter)

            loss_history.append(losses)

        print(loss_format.format(
            f"{epoch}",
            *[f"{l:.3f}" for l in np.mean(np.array(loss_history), axis=0)]
        ))

        scheduler_D.step()
        scheduler_Gin.step()
        scheduler_Gout.step()

        with torch.no_grad():
            discriminator.eval()
            if generators:
                plot_generator(generators, epoch)
            plot_discriminator(discriminator, epoch)
            discriminator.train()

except KeyboardInterrupt:
    # TODO: Checkpointing
    print(red("Interrupted", bold=True))

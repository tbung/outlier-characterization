from pathlib import Path
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import models
import data
from utils import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

c = config.Config()
c.parse_args()

writer = SummaryWriter(comment='_INN')

writer.add_hparams(c.__dict__, {})

checkpoints_path = Path(writer.get_logdir()) / 'checkpoints'
checkpoints_path.mkdir(exist_ok=True)


def tensor2imgs(t):
    imgrid = torchvision.utils.make_grid(t)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def build_z_fixed(latent_dim):
    z_fixed_t = np.zeros([latent_dim, latent_dim + 1])

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
    return torch.tensor(z_fixed, device=device, dtype=torch.float)


def plot_z_fixed():
    pass


def plot_random():
    pass


z_fixed = build_z_fixed(c.latent_dim)
t_prior = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(c.latent_dim, device=device),
    scale_tril=torch.eye(c.latent_dim, device=device)
)

model = models.INN_AA(c.latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)

n_iter = 0
for epoch in range(c.n_epochs):
    for samples, targets in tqdm(data.train_loader, leave=False,
                                 desc=f'{epoch:04d}'):
        optimizer.zero_grad()

        samples, targets = samples.to(device), targets.to(device)

        samples += c.add_noise_scale * torch.randn_like(samples, device=device)

        t, A, B = model(samples)

        zz = torch.sum(t**2, dim=1)
        jac = model.inn.log_jacobian(run_forward=False)

        neg_log_likeli = 0.5 * zz - jac

        nllikelihood = torch.mean(neg_log_likeli)
        at_loss = torch.mean(torch.norm(B @ A @ z_fixed - z_fixed, dim=1))
        recon_loss = torch.norm(model.sample(A, z_fixed) - samples)

        loss = (
            nllikelihood
            + c.at_loss_factor * at_loss
            + c.recon_loss_factor * recon_loss
        )

        loss.backward()
        optimizer.step()

        writer.add_scalar('Losses/Likelihood', nllikelihood, n_iter)
        # writer.add_scalar('Losses/Class', class_loss, n_iter)
        writer.add_scalar('Losses/Archetype', at_loss, n_iter)
        writer.add_scalar('Losses/Reconstruction', recon_loss, n_iter)

        n_iter += 1

    with torch.no_grad():
        recreated = model.sample(torch.eye(c.latent_dim + 1, device=device), z_fixed)
        writer.add_image('Z_fixed', tensor2imgs(recreated), epoch)
        # print(torch.argmax(labels, dim=1))

        x, y = next(iter(data.train_loader))
        x = x.to(device)
        t, A, B = model(x)
        recreated = model.sample(A, z_fixed)
        writer.add_image('Recreated', tensor2imgs(recreated[:64]), epoch)

        sample = model.inn(torch.randn(64, 28*28, device=device), rev=True)
        writer.add_image('Latent Sample', tensor2imgs(sample), epoch)

        if epoch % 10 == 0:
            latent_codes = torch.empty(0, c.latent_dim)
            # inn_codes = torch.empty(0, 28*28)
            labels = torch.empty(0, dtype=torch.long)
            for samples, targets in tqdm(data.train_loader, leave=False):
                samples = samples.to(device)
                t, A, B = model(samples)
                A_ = torch.sin(A * np.pi * 2/3)
                latent_codes = torch.cat([latent_codes, (A_ @ z_fixed /
                                                         np.sin(np.pi * 2/3)).cpu()], dim=0)
                # inn_codes = torch.cat([inn_codes, t.cpu()], dim=0)
                labels = torch.cat([labels, targets], dim=0)

            if c.latent_dim > 2:
                _, _, v = torch.svd(latent_codes)
                latent_codes = latent_codes @ v

            fig, ax = plt.subplots()
            img = ax.scatter(latent_codes[:, 0], latent_codes[:, 1], alpha=0.4,
                             c=labels, cmap='tab10', vmin=0, vmax=9)
            ax.scatter(z_fixed[:, 0].cpu(), z_fixed[:, 1].cpu(), marker='x', s=100, c='k')
            ax.set_aspect('equal')
            fig.colorbar(img)
            writer.add_figure('Latent Space', fig, epoch)

            # _, _, v = torch.svd(inn_codes)
            # inn_codes = inn_codes @ v

            # fig, ax = plt.subplots()
            # ax.scatter(inn_codes[:, 0], inn_codes[:, 1], alpha=0.4,
            #            c=labels, cmap='tab10', vmin=0, vmax=9)
            # ax.set_aspect('equal')
            # writer.add_figure('INN Space', fig, epoch)

    torch.save(model.state_dict(), checkpoints_path / 'innaa.pt')

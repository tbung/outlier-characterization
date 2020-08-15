from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import models
import data
from utils import config, logger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

c = config.Config()
c.parse_args()

writer = SummaryWriter(comment='_INN')
log = logger.LossLogger()

writer.add_hparams(c.__dict__, {})

checkpoints_path = Path(writer.get_logdir()) / 'checkpoints'
checkpoints_path.mkdir(exist_ok=True)


def tensor2imgs(t, n=8):
    imgrid = torchvision.utils.make_grid(t, n)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def build_z_fixed(latent_dim):
    z_fixed_t = np.zeros([latent_dim, latent_dim + 2])

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
                        requires_grad=True)


def plot_z_fixed():
    pass


def plot_random():
    pass


z_fixed = build_z_fixed(c.latent_dim)
t_prior = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(c.latent_dim, device=device),
    scale_tril=torch.eye(c.latent_dim, device=device)
)

model = models.INN_AA(c.latent_dim, interpolation=c.interpolation,
                      weight_norm_exp=c.weight_norm_exp,
                      weight_norm_constraint=c.weight_norm_constraint,
                      pretrained='../mnist/runs_backup/May13_19-39-26_GLaDOS/checkpoints/generator_in.pt').to(device)
optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad,
                                         model.parameters())) + [z_fixed], lr=c.lr)

fill = torch.zeros((c.ncl, c.ncl, c.img_width, c.img_width), device=device)
for i in range(c.ncl):
    fill[i, i, :, :] = 1


def make_cond(labels):
    cond_tensor = torch.zeros(labels.size(0), c.ncl).cuda()
    if c.conditional:
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.)
    else:
        cond_tensor[:, 0] = 1
    return cond_tensor


targets = torch.tensor(range(10), device=device).repeat_interleave(c.latent_dim
                                                                   + 2)
fixed_cond_z = [
    fill[:, :, :16, :16][targets],
    fill[:, :, :8, :8][targets],
    make_cond(targets)
]

targets = torch.tensor(range(10), device=device).repeat_interleave(10)
fixed_cond = [
    fill[:, :, :16, :16][targets],
    fill[:, :, :8, :8][targets],
    make_cond(targets)
]


n_iter = 0
for epoch in range(c.n_epochs):
    for samples, targets in tqdm(data.train_loader, leave=False,
                                 desc=f'{epoch:04d}'):
        optimizer.zero_grad()

        # z_fixed[-1] = 0 * z_fixed[-1]

        samples, targets = samples.to(device), targets.to(device)
        cond = [
            fill[:, :, :16, :16][targets],
            fill[:, :, :8, :8][targets],
            make_cond(targets)
        ]

        samples += c.add_noise_scale * torch.randn_like(samples, device=device)

        t, A, B = model(samples, cond)

        zz = torch.sum(t**2, dim=1)
        jac = model.inn.log_jacobian(run_forward=False)

        # neg_log_likeli = 0.5 * zz - jac

        recreated, sideinfo = model.sample(A, z_fixed, cond)

        # nllikelihood = torch.mean(neg_log_likeli)
        nllikelihood = torch.tensor(0, dtype=torch.float, device=device)
        sample_latent_mean = A @ z_fixed
        at_loss = torch.mean(torch.norm(B @ sample_latent_mean - z_fixed, dim=1))
        proto_loss = torch.sum((z_fixed[-1] - torch.mean(sample_latent_mean,
                                                         dim=0))**2)
        recon_loss = torch.norm(recreated - samples)
        class_loss = F.cross_entropy(sideinfo, targets)

        loss = (
            nllikelihood
            + c.at_loss_factor * at_loss
            + c.recon_loss_factor * recon_loss
            + c.class_loss_factor * class_loss
            + 10 * c.at_loss_factor * proto_loss
        )

        loss.backward()
        optimizer.step()

        log.add_loss('nll', nllikelihood)
        log.add_loss('at', at_loss)
        log.add_loss('rec', recon_loss)
        log.add_loss('class', class_loss)
        log.add_loss('proto', proto_loss)

        writer.add_scalar('Losses/Likelihood', nllikelihood, n_iter)
        writer.add_scalar('Losses/Class', class_loss, n_iter)
        writer.add_scalar('Losses/Archetype', at_loss, n_iter)
        writer.add_scalar('Losses/Reconstruction', recon_loss, n_iter)

        n_iter += 1

    with torch.no_grad():
        recreated, _ = model.sample(
            torch.eye(c.latent_dim + 2, device=device).repeat(10, 1),
            z_fixed, fixed_cond_z
        )
        writer.add_image('Z_fixed', tensor2imgs(recreated, c.latent_dim + 2), epoch)
        # print(torch.argmax(labels, dim=1))

        x, y = next(iter(data.train_loader))
        x, y = x.to(device), y.to(device)
        cond = [
            fill[:, :, :16, :16][y],
            fill[:, :, :8, :8][y],
            make_cond(y)
        ]
        t, A, B = model(x, cond)
        recreated, sideinfo = model.sample(A, z_fixed, cond)
        writer.add_image('Recreated', tensor2imgs(recreated[:64]), epoch)
        writer.add_scalar(
            'Losses/Sample_Class_Accuracy',
            (F.softmax(sideinfo, dim=1).max(dim=1)[1] == y).float().mean(),
            epoch
        )

        sample = model.inn(torch.randn(100, 32*32, device=device), fixed_cond, rev=True)
        writer.add_image('Latent Sample', tensor2imgs(sample, 10), epoch)

        if epoch % 10 == 0:
            latent_codes = torch.empty(0, c.latent_dim)
            # inn_codes = torch.empty(0, 28*28)
            labels = torch.empty(0, dtype=torch.long)
            for samples, targets in tqdm(data.train_loader, leave=False):
                samples, targets = samples.to(device), targets.to(device)
                cond = [
                    fill[:, :, :16, :16][targets],
                    fill[:, :, :8, :8][targets],
                    make_cond(targets)
                ]
                t, A, B = model(samples, cond)
                if c.interpolation == "linear":
                    latent_codes = torch.cat([latent_codes, (A @ z_fixed).cpu()], dim=0)
                elif c.interpolation == "slerp":
                    A_ = torch.sin(A * np.pi * 2/3)
                    latent_codes = torch.cat([latent_codes, (A_ @ z_fixed /
                                                             np.sin(np.pi * 2/3)).cpu()], dim=0)
                # inn_codes = torch.cat([inn_codes, t.cpu()], dim=0)
                labels = torch.cat([labels, targets.cpu()], dim=0)

            try:
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
                writer.flush()
            except Exception:
                writer.flush()

            # _, _, v = torch.svd(inn_codes)
            # inn_codes = inn_codes @ v

            # fig, ax = plt.subplots()
            # ax.scatter(inn_codes[:, 0], inn_codes[:, 1], alpha=0.4,
            #            c=labels, cmap='tab10', vmin=0, vmax=9)
            # ax.set_aspect('equal')
            # writer.add_figure('INN Space', fig, epoch)

    log.flush()
    torch.save(model.state_dict(), checkpoints_path / 'innaa.pt')
    torch.save(z_fixed, checkpoints_path / 'z_fixed.pt')

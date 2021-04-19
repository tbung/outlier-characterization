from pathlib import Path
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import models
import data

device = "cuda" if torch.cuda.is_available() else "cpu"

writer = SummaryWriter()

checkpoints_path = Path(writer.get_logdir()) / "checkpoints"
checkpoints_path.mkdir(exist_ok=True)

latent_dim = 7
nAT = latent_dim + 1
n_epochs = 5000
# DAA loss: weights
at_loss_factor = 15.0  # 80.0
class_loss_factor = 5.0  # 80.0
recon_loss_factor = 10.0  # 1.0
kl_loss_factor = 15.0  # 40.0


def tensor2imgs(t, n=8):
    imgrid = torchvision.utils.make_grid(t, n)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def build_z_fixed():
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


z_fixed = build_z_fixed()
t_prior = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(latent_dim, device=device),
    scale_tril=torch.eye(latent_dim, device=device),
)

model = models.DeepAA(z_fixed, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_iter = 0
for epoch in range(n_epochs):
    for samples, targets in tqdm(data.train_loader, leave=False):
        optimizer.zero_grad()

        samples, targets = samples.to(device), targets.to(device)

        z_pred, mu, sigma, t_posterior = model.encode(samples)
        x_hat, sideinfo = model.decode(t_posterior.rsample())

        # losses
        divergence = torch.mean(
            torch.distributions.kl.kl_divergence(t_posterior, t_prior)
        )

        at_loss = torch.mean(torch.norm(z_pred - z_fixed, dim=1))

        class_loss = torch.nn.functional.nll_loss(sideinfo, targets)

        likelihood = torch.mean(x_hat.log_prob(samples))

        elbo = -torch.mean(
            recon_loss_factor * likelihood
            - class_loss_factor * class_loss
            - at_loss_factor * at_loss
            - kl_loss_factor * divergence
        )

        elbo.backward()
        optimizer.step()

        writer.add_scalar("Losses/Likelihood", likelihood, n_iter)
        writer.add_scalar("Losses/Class", class_loss, n_iter)
        writer.add_scalar("Losses/Archetype", at_loss, n_iter)
        writer.add_scalar("Losses/Divergence", divergence, n_iter)

        n_iter += 1

    with torch.no_grad():
        recreated, labels = model.decode(z_fixed)
        writer.add_image("Z_fixed", tensor2imgs(recreated.mean, 5), epoch)
        print(torch.argmax(labels, dim=1))

        recreated, labels = model.decode(torch.zeros(1, latent_dim, device=device))
        writer.add_image("Max_Likeli", tensor2imgs(recreated.mean, 5), epoch)

        if epoch % 10 == 0:
            latent_codes = torch.empty(0, latent_dim)
            labels = torch.empty(0, dtype=torch.long)
            for samples, targets in tqdm(data.train_loader, leave=False):
                samples = samples.to(device)
                _, mu, _, _ = model.encode(samples)
                latent_codes = torch.cat([latent_codes, mu.cpu()], dim=0)
                labels = torch.cat([labels, targets], dim=0)

            if latent_dim > 2:
                _, _, v = torch.svd(latent_codes)
                latent_codes = latent_codes @ v

            fig, ax = plt.subplots()
            ax.scatter(
                latent_codes[:, 0],
                latent_codes[:, 1],
                alpha=0.4,
                c=labels,
                cmap="tab10",
                vmin=0,
                vmax=9,
            )
            ax.set_aspect("equal")
            writer.add_figure("Latent Space", fig, epoch)

    torch.save(model.state_dict(), checkpoints_path / "deepaa.pt")

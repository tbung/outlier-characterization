from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from crayons import red

import data
import models
import evaluation
from utils import config, logger


device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    # Load config and set up logging and stuff
    c = config.Config()
    c.parse_args()

    writer = SummaryWriter(comment=f"_{c.model_type}")
    log = logger.LossLogger()

    checkpoints_path = Path(writer.get_logdir()) / "checkpoints"
    checkpoints_path.mkdir(exist_ok=True)

    c.save(checkpoints_path / "config.toml")

    writer.add_hparams(c.__dict__, {})

    # Prepare model, dataset and any other things we need during training
    dataset_train = data.get_dataset(c.dataset, train=True)

    model = models.get_model(c.model_type)(**c.__dict__)
    model.to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=c.lr,
        betas=[0.9, 0.999],
        weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, c.n_epochs // 3, c.lr_step)

    # Prepare constants for plotting, etc.
    fixed_noise_inn = torch.randn(
        c.n_classes * c.n_classes, c.n_channels * c.img_width * c.img_width, device=device
    )

    fixed_labels = torch.tensor(
        list(range(c.n_classes)), device=device
    ).repeat_interleave(c.n_classes)

    fixed_cond = model.labels2condition(fixed_labels)

    n_archetypes = c.latent_dim + (2 if c.use_proto_z else 1)
    fixed_cond_z = model.labels2condition(
        torch.tensor(range(c.n_classes), device=device).repeat_interleave(n_archetypes)
    )

    try:  # Catch KeyboardInterrupts to do checkpointing
        for epoch in range(c.n_epochs):
            for n, (samples, labels) in enumerate(
                tqdm(
                    dataset_train,
                    leave=False,
                    mininterval=1.0,
                    ncols=80,
                    dynamic_ncols=True,
                )
            ):

                n_iter = n + (epoch * len(dataset_train))

                samples, labels = samples.to(device), labels.to(device)

                samples += c.add_image_noise * torch.randn_like(samples, device=device)

                optimizer.zero_grad()
                losses = model.compute_losses(samples, labels)
                total_loss = torch.tensor(0)
                for key, loss in losses.items():
                    total_loss = total_loss + loss
                    writer.add_scalar(f"Loss/{key}", loss, n_iter)
                    log.add_loss(f"{key}", loss)

                total_loss.backward()

                optimizer.step()

            with torch.no_grad():
                # Plot latent sample
                if c.latent_dist == "mixture":
                    if c.model_type == "INN":
                        latent = model(samples, model.labels2condition(labels))
                    else:
                        latent = model.inn(samples, model.labels2condition(labels))

                    mean = torch.zeros(
                        c.n_classes, latent.shape[1], dtype=torch.float, device=device
                    )
                    var = torch.zeros(
                        c.n_classes, latent.shape[1], dtype=torch.float, device=device
                    )
                    for i in range(c.n_classes):
                        mean[i] = latent[labels == i].mean(dim=0)
                        var[i] = latent[labels == i].var(dim=0)
                    fixed_noise_inn = torch.normal(
                        mean[fixed_labels], var[fixed_labels]
                    )

                if c.model_type == "INN":
                    samples = model.sample(fixed_noise_inn, fixed_cond)
                else:
                    samples = model.inn.sample(fixed_noise_inn, fixed_cond)
                writer.add_image(
                    "Samples/In-Distribution", data.tensors2image(c.dataset, samples, c.n_classes), epoch
                )

                if c.model_type == "INN_AA":
                    # Plot z_arch
                    samples, _ = model.sample(
                        torch.eye(n_archetypes, device=device).repeat(c.n_classes, 1),
                        fixed_cond_z,
                    )
                    writer.add_image(
                        "Z_fixed", data.tensors2image(c.dataset, samples, n_archetypes), epoch
                    )

                    # TODO: Plot archetype sample

                    # Plot recreation
                    x, y = next(iter(dataset_train))
                    x, y = x.to(device), y.to(device)
                    t, A, B = model(x, model.labels2condition(y))
                    recreated, sideinfo = model.sample(A, model.labels2condition(y))
                    writer.add_image("Recreated", data.tensors2image(c.dataset, recreated[:64]), epoch)
                    z_pred = B @ (
                        torch.einsum(
                            "bj, bjk -> bk",
                            A,
                            model.z_arch[y],
                        )
                        if c.z_per_class
                        else torch.einsum("bj, jk -> bk", A, model.z_arch)
                    )

                if epoch % 10 == 0:
                    # Plot latent space projection
                    latent_rs, labels, gradients = evaluation.compute_latent(
                        model, dataset_train, compute_grad=True
                    )

                    # Try to do PCA, ignore if it fails
                    try:
                        if c.latent_dim > 2:
                            _, _, v = torch.svd(latent_rs)
                            latent_rs = latent_rs @ v

                        fig, ax = plt.subplots()
                        img = ax.scatter(
                            latent_rs[:, 0],
                            latent_rs[:, 1],
                            alpha=0.4,
                            c=labels,
                            cmap="tab10",
                            vmin=0,
                            vmax=9,
                        )
                        if c.model_type == "INN_AA":
                            if c.z_per_class:
                                ax.scatter(
                                    model.z_arch[:, :, 0].cpu(),
                                    model.z_arch[:, :, 1].cpu(),
                                    marker="x",
                                    s=100,
                                    c="k",
                                )
                            else:
                                ax.scatter(
                                    model.z_arch[:, 0].cpu(),
                                    model.z_arch[:, 1].cpu(),
                                    marker="x",
                                    s=100,
                                    c="k",
                                )
                            ax.scatter(
                                z_pred[:, 0].cpu(),
                                z_pred[:, 1].cpu(),
                                marker="x",
                                s=100,
                                c="r",
                            )
                        ax.set_aspect("equal")
                        fig.colorbar(img)
                        writer.add_figure("Latent Space", fig, epoch)
                        writer.add_scalar(
                            "Grad output wrt params", gradients.mean(), epoch
                        )
                        writer.flush()
                    except ValueError:
                        writer.flush()

            log.flush()
            scheduler.step()

            if epoch % 10 == 0:
                model.save(checkpoints_path / f"{c.model_type}_{epoch}.pt")
            if epoch % 100 == 0:
                # If we reach a multiple of 100 epoch, delete checkpoints that
                # aren't from multiples of 100 to save disk space
                for p in checkpoints_path.glob('INN*_?[1-9]*.pt'):
                    p.unlink()

    except KeyboardInterrupt:
        print(red("Interrupted", bold=True))

import torch
import numpy as np
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_latent(
    model,
    data_loader,
    compute_grad=False,
    fixed_condition=None,
    tag="",
    save=False,
    model_path=None,
):
    """Compute latent representation of data and write to disk"""
    with torch.set_grad_enabled(compute_grad):
        latent_rs = torch.empty(0, 32 * 32)
        labels = torch.empty(0).long()
        gradients = torch.empty(0)

        for samples, slabels in tqdm(data_loader, leave=False):
            samples = samples.to(device)
            slabels = slabels.to(device)
            slabels_ = (
                slabels
                if fixed_condition is None
                else torch.full_like(slabels, fixed_condition, device=device)
            )

            if str(model).startswith("INN("):
                output = model(samples, model.labels2condition(slabels_))
            else:
                t, A, B = model(samples, model.labels2condition(slabels_))
                if model.interpolation == "linear":
                    output = (
                        torch.einsum(
                            "bj, bjk -> bk",
                            A,
                            model.z_arch[slabels],
                        )
                        if model.z_per_class
                        else torch.einsum("bj, jk -> bk", A, model.z_arch)
                    ).cpu()
                elif model.interpolation == "slerp":
                    A_ = torch.sin(A * np.pi * 2 / 3)
                    output = (
                        torch.einsum(
                            "bj, bjk -> bk",
                            A_,
                            model.z_arch[slabels],
                        )
                        / np.sin(np.pi * 2 / 3)
                        if model.z_per_class
                        else torch.einsum("bj, jk -> bk", A_, model.z_arch)
                        / np.sin(np.pi * 2 / 3)
                    ).cpu()
                else:
                    raise ValueError("Cannot understand archetype interpolation method")

            latent_rs = torch.cat([latent_rs, output.detach().cpu()])
            labels = torch.cat([labels, slabels.detach().cpu()])

            if compute_grad:
                model.zero_grad()
                # Since autograd computes the sum of the gradients and we want to
                # see the 2-norm of the gradient, we cube the output before doing
                # backprop
                (output ** 3 / 3).backward(torch.ones_like(output))

                grad = torch.tensor(0.0)
                n_params = 0
                for param in model.parameters():
                    grad += param.grad.sum().cpu()
                    n_params += param.nelement()

                grad /= n_params * output.nelement()
                grad = torch.sqrt(grad)
                gradients = torch.cat([gradients, grad.detach().cpu().reshape(1, 1)])

    if save:
        if model_path is None:
            raise ValueError("You need to provide a path to a folder to save to disk")
        torch.save(
            latent_rs,
            model_path
            / "checkpoints/latent{}{}.pt".format(
                tag, "" if fixed_condition is None else f"_{fixed_condition}"
            ),
        )
        torch.save(
            labels,
            model_path
            / "checkpoints/classes{}{}.pt".format(
                tag, "" if fixed_condition is None else f"_{fixed_condition}"
            ),
        )
        if compute_grad:
            torch.save(
                gradients,
                model_path
                / "checkpoints/grad{}{}.pt".format(
                    tag, "" if fixed_condition is None else f"_{fixed_condition}"
                ),
            )

    return latent_rs, labels, gradients

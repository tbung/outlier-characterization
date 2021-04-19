import torch
import numpy as np
from tqdm import tqdm
from lib import data, models
from utils import config
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import dirichlet
from scipy.spatial import Delaunay

import matplotlib
matplotlib.use("agg")

import matplotlib.pyplot as plt
import matplotlib.tri as tri


device = "cuda" if torch.cuda.is_available() else "cpu"
tmp_path = Path("./tmpdata")


def load_model(run_path):
    model_path = Path(run_path)
    c = config.Config()
    c.load(model_path / "checkpoints/config.toml")
    c.load_inn = model_path / "checkpoints/latest.pt"
    model = models.get_model(c.model_type)(**c.__dict__)
    model.to(device)

    return model, c


# TODO: Do we need this? Also rework naming files and where they go
def compute_latent(
    model,
    dataset,
    compute_grad=False,
    fixed_condition=None,
    save=False,
):
    """Compute latent representation of data and write to disk"""
    data_loader = data.get_dataset(dataset)
    # print(next(iter(data_loader)))
    with torch.set_grad_enabled(compute_grad):
        latent_rs = torch.empty(0, model.latent_dim)
        labels = torch.empty(0).long()
        gradients = torch.empty(0)

        for samples, slabels in tqdm(data_loader, leave=False):
            samples = samples.to(device)
            samples.requires_grad = True
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
                # output.backward(torch.ones_like(output))

                # grad = torch.tensor(0.0)
                # n_params = 0
                # for param in model.parameters():
                #     grad += (param.grad.data.cpu()/output.nelement()).norm(2)

                grad = torch.autograd.grad(
                    output, samples, torch.ones_like(output, requires_grad=True)
                )[0].norm(2, dim=(-3, -2, -1))

                gradients = torch.cat([gradients, grad.detach().cpu().reshape(512, 1)])

    if save:
        (tmp_path / dataset).mkdir(parents=True, exist_ok=True)
        torch.save(
            latent_rs,
            tmp_path
            / dataset
            / "latent{}.pt".format(
                "" if fixed_condition is None else f"_{fixed_condition}"
            ),
        )
        torch.save(
            labels,
            tmp_path
            / dataset
            / "classes{}.pt".format(
                "" if fixed_condition is None else f"_{fixed_condition}"
            ),
        )
        if compute_grad:
            torch.save(
                gradients,
                tmp_path
                / dataset
                / "grad{}.pt".format(
                    "" if fixed_condition is None else f"_{fixed_condition}"
                ),
            )

    return latent_rs, labels, gradients


def compute_latent_all(inn):
    pass


class LDA:
    def __init__(self, n_components=2, tol=1e-4):
        self.n_components = n_components
        self.tol = tol

    def fit(self, X, y):
        """SVD solver.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        """
        n_samples, n_features = X.shape
        device = X.device

        self.classes_, y_t, counts = torch.unique(
            y, return_inverse=True, return_counts=True
        )

        n_classes = len(self.classes_)

        self.means_ = torch.zeros((len(self.classes_), X.shape[1]), device=device)
        self.means_.index_add_(0, y_t, X)
        self.means_ /= counts[:, None]

        self.priors_ = counts / float(len(y))

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        self.xbar_ = self.priors_ @ self.means_

        Xc = torch.cat(Xc, dim=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = Xc.std(dim=0, unbiased=False)
        # avoid division by zero in normalization
        std[std == 0] = 1.0
        fac = 1.0 / (n_samples - n_classes)

        # 2) Within variance scaling
        X = np.sqrt(fac) * (Xc / std)

        # SVD of centered (within)scaled data
        # U, S, V = torch.linalg.svd(X, full_matrices=False)
        U, S, V = torch.svd(X, some=True)
        V = V.T

        rank = torch.sum(S > self.tol)
        # Scaling of within covariance is: V' 1/S
        scalings = (V[:rank] / std).T / S[:rank]

        # 3) Between variance scaling
        # Scale weighted centers
        X = (
            (torch.sqrt((n_samples * self.priors_) * fac))
            * (self.means_ - self.xbar_).T
        ).T @ scalings

        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        # _, S, V = torch.linalg.svd(X, full_matrices=False)
        _, S, V = torch.svd(X, some=True)
        V = V.T

        rank = torch.sum(S > self.tol * S[0])
        self.scalings_ = scalings @ V.T[:, :rank]

    def transform(self, X):
        X_new = (X - self.xbar_) @ self.scalings_
        return X_new[:, : self.n_components]


def plot_distance_matrix(dataset):
    class_labels = data.get_dataset(dataset).dataset.classes

    mnist_latent = torch.load(tmp_path / "EMNIST/latent.pt")[:20000]
    mnist_classes = torch.load(tmp_path / "EMNIST/classes.pt")[:20000]

    d_matrix = torch.zeros((10, len(class_labels)))

    for i in tqdm(range(10)):

        other_latent = torch.load(tmp_path / f"{dataset}/latent_{i}.pt")[:50000]
        other_classes = torch.load(tmp_path / f"{dataset}/classes_{i}.pt")[:50000]

        lda = LDA()

        all_data = torch.cat([mnist_latent[mnist_classes == i], other_latent], dim=0)
        all_targets = torch.cat(
            [
                i
                * torch.ones(
                    (mnist_latent[mnist_classes == i].shape[0],), dtype=torch.long
                ),
                10 + other_classes,
            ]
        )
        lda.fit(all_data.cuda(), all_targets.cuda())

        mnist_latent_t = lda.transform(mnist_latent[mnist_classes == i][:1000].cuda())[
            :, :2
        ]
        other_latent_t = lda.transform(other_latent.cuda())[:, :2]

        for j in range(len(class_labels)):
            d_matrix[i, j] = torch.norm(
                mnist_latent_t - other_latent_t[other_classes == j][:1000], dim=1
            ).mean()

    fig, ax = plt.subplots(figsize=(22, 22))
    m = ax.matshow(d_matrix)
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticks(range(10))
    ax.set_title("Mean Cluster Distance between MNIST and Letters")
    fig.colorbar(m, orientation="horizontal")
    return fig, ax


def plot_lda_clusters():
    pass


def plot_radius_hist():
    pass


def plot_single_increasing_distance(model, dataset, n=8, min=0.5, max=3, c=0):
    s = np.linspace(min, max, n)
    t = []
    for i in s:
        t.append(torch.normal(torch.zeros(1, 3 * 1024), i * torch.ones(1, 3 * 1024)))

    t = torch.cat(t, dim=0)
    t = t.cuda()
    labels = torch.full((n,), c, dtype=torch.long, device="cuda")
    samples = model.sample(t, model.labels2condition(labels))

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})
    fig, ax = plt.subplots(tight_layout=True)
    ax.imshow(data.tensors2image(dataset, samples).transpose(1, 2, 0))
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks(list(range(17, n * 34, 34)))
    ax.set_xticklabels(s.round(1))
    ax.set_yticks([])

    return fig, ax


def plot_aa_dist_with_sample(model, dataset, n_classes, s1, s2, s3):
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    dim = 2
    # z = build_z_simplex(dim, False)
    z = model.z_arch.cpu().detach().numpy()
    fixed_cond_z = model.labels2condition(
        torch.tensor(range(n_classes), device=device).repeat_interleave(n_classes)
    )

    A = np.random.dirichlet(np.array([s1, s2, s3]), 1000)
    A = A / A.sum(axis=1, keepdims=True)

    phi = np.linspace(0, 2 * np.pi)
    with torch.no_grad():
        samples, _ = model.sample(
            torch.tensor(
                A[: (n_classes * n_classes)], device=device, dtype=torch.float
            ),
            fixed_cond_z,
        )

        triangle = tri.Triangulation(z[:, 0], z[:, 1])

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=4)
    t = Delaunay(z)
    p = np.stack((trimesh.x, trimesh.y), axis=1)
    b = t.transform[0, :2].dot(np.transpose(p - t.transform[0, 2]))

    b = np.clip(np.c_[np.transpose(b), 1 - b.sum(axis=0)], 1e-16, 1)
    pvals = [dirichlet.pdf(x, np.array([s1, s2, s3])) for x in b]

    tcm = axes[0].tricontourf(trimesh, pvals, 200, cmap="plasma", alpha=1)

    t = A @ z
    axes[0].scatter(t[:, 0], t[:, 1], c="k", alpha=0.1)
    axes[0].scatter(z[:, 0], z[:, 1], marker="x", c="r", alpha=1)

    axes[0].set_aspect("equal")
    axes[0].set_xlim((-1.6, 2.2))
    axes[0].set_ylim((-1.6, 2.2))
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    axes[1].imshow(data.tensors2image(dataset, samples, n_classes).transpose(1, 2, 0))
    axes[1].set_axis_off()

    return fig, axes


if __name__ == "__main__":
    model_path = Path("runs/Jan22_21-08-39_GLaDOS_INN/")
    c = config.Config()
    c.load(model_path / "checkpoints/config.toml")
    c.load_inn = model_path / "checkpoints/INN_1490.pt"
    inn = models.INN(**c.__dict__).to(device)
    fig, _ = plot_single_increasing_distance(inn, c.dataset)
    fig.savefig("test.png")

    c = config.Config()
    # c.load("./runs/Oct23_01-37-19_GLaDOS_INN_AA/checkpoints/config.toml")
    c.load("./runs/Dec09_01-26-04_GLaDOS_INN_AA/checkpoints/config.toml")
    # c.load("./runs/Dec09_01-39-52_GLaDOS_INN_AA/checkpoints/config.toml")
    # c.load("./runs/Dec09_02-52-36_GLaDOS_INN_AA/checkpoints/config.toml")
    # c.load("./runs/Sep03_23-38-37_GLaDOS_INN_AA/checkpoints/config.toml")
    # c.load("./runs/Dec22_23-45-54_GLaDOS_INN_AA/checkpoints/config.toml")

    model = models.INN_AA(**c.__dict__)
    # model.load("./runs/Oct23_01-37-19_GLaDOS_INN_AA/checkpoints/INN_AA_490.pt")
    model.load("./runs/Dec09_01-26-04_GLaDOS_INN_AA/checkpoints/INN_AA_300.pt")
    # model.load("./runs/Dec09_01-39-52_GLaDOS_INN_AA/checkpoints/INN_AA_300.pt")
    # model.load("./runs/Dec09_02-52-36_GLaDOS_INN_AA/checkpoints/INN_AA_300.pt")
    # model.load("./runs/Sep03_23-38-37_GLaDOS_INN_AA/checkpoints/INN_AA.pt")
    # model.load("./runs/Dec22_23-45-54_GLaDOS_INN_AA/checkpoints/INN_AA_700.pt")
    model.to("cuda")

    # fig, _ = plot_aa_dist_with_sample(model, c.dataset, c.n_classes, 3, 3, 0.1)
    # fig.savefig("test.png")

    # fig_pt, _ = plot_distance_matrix("letters")
    # fig_np, _ = plot_distance_matrix_np("letters")

    # fig_pt.savefig("test_pt.png")
    # fig_np.savefig("test_np.png")

    # for i in range(100):
    #     # Test LDA implementation
    #     x, y = torch.rand((1000, 1024)), torch.randint(10, (1000,))
    #     lda_sk = LinearDiscriminantAnalysis(n_components=2)
    #     lda_pt = LDA()
    #     lda_sk.fit(x, y)
    #     lda_pt.fit(x.cuda(), y.cuda())
    #     assert np.allclose(lda_sk.priors_, lda_pt.priors_.cpu())
    #     assert np.allclose(lda_sk.means_, lda_pt.means_.cpu())
    #     # print(lda_sk.scalings_.shape)
    #     # print(lda_pt.scalings_.shape)
    #     # print(np.abs(lda_sk.xbar_ - lda_pt.xbar_.cpu().numpy()).max())
    #     # print(np.abs(lda_sk.scalings_ - lda_pt.scalings_.cpu().numpy()).max())
    #     assert np.allclose(lda_sk.xbar_, lda_pt.xbar_.cpu())
    #     # assert np.allclose(lda_sk.scalings_, lda_pt.scalings_.cpu())
    #     x_sk = lda_sk.transform(x)
    #     x_pt = lda_pt.transform(x.cuda())
    #     # print(x_sk[:10])
    #     # print(x_pt[:10])
    #     a = np.abs(np.abs(x_sk) - np.abs(x_pt.cpu().numpy()))
    #     b = 1e-8 + 1e-1 * np.abs(x_pt.cpu().numpy())
    #     print(a[a>b])
    #     print(b[a>b])
    #     print(x_sk[a>b])
    #     print(x_pt[a>b])
    #     print()
    #     # print(i)
    #     # assert torch.allclose(torch.abs(torch.tensor(x_sk)), torch.abs(x_pt.cpu()), rtol=1e-2)

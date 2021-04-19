from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

import numpy as np

from tqdm import tqdm
from crayons import red, cyan

import data
from utils import config
from logger import LossLogger
from models import Classifier, Generator, Discriminator, INN


CHECKPOINTS_PATH = Path("./checkpoints")
CHECKPOINTS_PATH.mkdir(exist_ok=True)

writer = SummaryWriter()
logger = LossLogger()

checkpoints_path = Path(writer.get_logdir()) / "checkpoints"
checkpoints_path.mkdir(exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

c = config.Config()
c.parse_args()

writer.add_hparams(c.__dict__, {})


def tensor2imgs(t):
    imgrid = torchvision.utils.make_grid(t, c.ncl)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def init_inn(mod):
    for key, param in mod.named_parameters():
        split = key.split(".")
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[3][-1] == "2":  # last convolution in the coeff func
                param.data.fill_(0.0)


def make_cond(labels):
    cond_tensor = torch.zeros(labels.size(0), c.ncl).cuda()
    if c.conditional:
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.0)
    else:
        cond_tensor[:, 0] = 1
    return cond_tensor


def pretrain_classifier(restore=True):
    print(cyan("Pretraining Classifier", bold=True))
    if restore and (CHECKPOINTS_PATH / "classifier_mnist.pth").is_file():
        return torch.load(CHECKPOINTS_PATH / "classifier_mnist.pth")

    model = Classifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_classifier)

    for epoch in range(c.n_epochs_pretrain):
        model.train()
        confidence = []
        for i, (x, y) in enumerate(tqdm(data.train_loader)):
            n_iter = i + epoch * len(data.train_loader)
            x = x.to(device)
            y = y.to(device)

            # x += c.add_image_noise * torch.randn_like(x, device=device)

            noise = torch.randn_like(x).to(device)

            uniform_dist = torch.Tensor(x.size(0), c.ncl).fill_((1 / c.ncl)).to(device)

            optimizer.zero_grad()

            output = model(x)
            confidence.append(F.softmax(output, dim=1).max(dim=1)[0].mean().item())
            loss = F.cross_entropy(output, y)
            kl_loss = F.kl_div(
                F.log_softmax(model(noise), dim=1), uniform_dist, reduction="batchmean"
            )

            if n_iter % c.log_interval == 0:
                writer.add_scalar("Classifier/Cross Entropy", loss, n_iter)
                writer.add_scalar("Classifier/KL Div", kl_loss, n_iter)

            loss += c.beta_classifier * c.ncl * kl_loss
            loss.backward()

            optimizer.step()

        print(f"In-Dist Confidence: {np.mean(confidence)}")

        confidence = []
        model.eval()
        test_error = 0
        for x, y in data.test_loader:
            x = x.to(device)
            y = y.to(device)
            output = F.softmax(model(x), dim=1)
            confidence.append(output.max(dim=1)[0].mean().item())
            output = torch.argmax(output, dim=1)
            test_error += (output != y).sum() / y.size(0)

        print(f"Test Error: {test_error/len(data.test_loader):.5f}")
        noise_output = F.softmax(model(noise), dim=1).max().item()
        print(f"Test In-Dist Confidence: {np.mean(confidence)}")
        print(f"OO-Dist Confidence: {noise_output}")

    torch.save(model, CHECKPOINTS_PATH / "classifier_mnist.pth")

    return model


def update_discriminator_fake(
    disc, gen, disc_targets, gen_targets, noise_dim, **kwargs
):
    """Updates discriminator grads with fake samples

    Returns zero and does nothing if we train without a discriminator
    """
    if not c.use_discriminator:
        return torch.tensor(np.nan, dtype=torch.float)

    noise = torch.randn(noise_dim, device=device)
    fake_samples = gen(noise, gen_targets, **kwargs)
    output = disc(fake_samples.detach(), disc_targets).reshape(-1)
    errD_fake = F.binary_cross_entropy_with_logits(output, zeros)
    errD_fake.backward()
    return errD_fake.data


def update_generator_gan(
    gen, disc, gen_targets, disc_targets, noise, ignore_nan=True, **kwargs
):

    fake_samples = gen(noise, gen_targets, **kwargs)
    output = discriminator(fake_samples, disc_targets).reshape(-1)
    errG = F.binary_cross_entropy_with_logits(output, ones)

    if torch.isnan(errG).any():
        if ignore_nan:
            return errG
        else:
            raise ValueError("NaN in discriminator output")

    errG.backward()
    return errG


def update_generator_nll():
    pass


def update_generator_kl():
    pass


if c.train_inner_gan:
    generator_in = Generator(conditional=c.conditional).to(device)
    generator_in.apply(weights_init)
    optimizer_Gin = torch.optim.Adam(
        generator_in.parameters(), lr=c.lr_generator, betas=[0.5, 0.999]
    )
    scheduler_Gin = torch.optim.lr_scheduler.StepLR(
        optimizer_Gin, c.lr_step, c.lr_decay ** (1 / c.n_epochs)
    )
elif c.train_inner_inn:
    generator_in = INN().to(device)
    # init_inn(generator_in)
    optimizer_Gin = torch.optim.Adam(
        generator_in.parameters(),
        lr=c.lr_generator,
        betas=[0.9, 0.999],
        weight_decay=1e-5,
    )
    scheduler_Gin = torch.optim.lr_scheduler.StepLR(optimizer_Gin, c.lr_step, c.lr_step)
if c.train_outer_gan:
    generator_out = Generator(conditional=c.conditional).to(device)
    generator_out.apply(weights_init)
    optimizer_Gout = torch.optim.Adam(
        generator_out.parameters(), lr=c.lr_generator, betas=[0.5, 0.999]
    )
    scheduler_Gout = torch.optim.lr_scheduler.StepLR(
        optimizer_Gout, c.lr_step, c.lr_decay
    )
elif c.train_outer_inn:
    generator_out = INN().to(device)
    init_inn(generator_out)
    optimizer_Gout = torch.optim.Adam(
        generator_out.parameters(),
        lr=c.lr_generator,
        betas=[0.9, 0.999],
        weight_decay=1e-5,
    )
    scheduler_Gout = torch.optim.lr_scheduler.StepLR(
        optimizer_Gout, c.lr_step, c.lr_step
    )

if c.use_discriminator:
    discriminator = Discriminator(conditional=c.conditional).to(device)
    discriminator.apply(weights_init)
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=c.lr_discriminator, betas=[0.5, 0.999]
    )
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, c.lr_step, c.lr_decay)

if c.pretrain_classifier:
    classifier = pretrain_classifier(c.restore)
else:
    classifier = Classifier().to(device)


# Fixed noise to create test samples from
fixed_noise = torch.randn(100, c.nz, 1, 1, device=device)
fixed_noise_inn = torch.randn(100, c.nch * c.img_width * c.img_width, device=device)

fixed_targets = torch.tensor(list(range(c.ncl)), device=device).repeat_interleave(c.ncl)
onehot = torch.zeros(c.ncl, c.ncl)
onehot = onehot.scatter_(
    1, torch.LongTensor(list(range(c.ncl))).view(c.ncl, 1), 1
).view(c.ncl, c.ncl, 1, 1)
onehot = onehot.to(device)
fill = torch.zeros((c.ncl, c.ncl, c.img_width, c.img_width), device=device)
for i in range(c.ncl):
    fill[i, i, :, :] = 1

ones = torch.ones((c.batch_size,), device=device).float()
zeros = torch.zeros((c.batch_size,), device=device).float()
uniform_dist = torch.Tensor(c.batch_size, c.ncl).fill_((1 / c.ncl)).to(device)

ood_iter = iter(data.letter_loader)


print(cyan("Begin GAN Training", bold=True))
try:

    for epoch in range(c.n_epochs):

        for n, (samples, targets) in enumerate(
            tqdm(data.train_loader, leave=False, mininterval=1.0)
        ):
            n_iter = n + (epoch * len(data.train_loader))

            samples = samples.to(device, non_blocking=True)
            samples += c.add_image_noise * torch.randn_like(samples, device=device)
            targets = targets.to(device, non_blocking=True)

            labels = fill[targets]
            cond = [
                fill[:, :, :16, :16][targets],
                fill[:, :, :8, :8][targets],
                make_cond(targets),
            ]

            fake_targets = torch.randint(c.ncl, (samples.size(0),), device=device)
            fake_labels = fill[fake_targets]
            fake_targets_onehot = onehot[fake_targets]
            fake_cond = make_cond(fake_targets)

            ###
            # Update Discriminator
            ###

            if c.use_discriminator:

                optimizer_D.zero_grad()

                output = discriminator(samples, labels).reshape(-1)
                errD_real = F.binary_cross_entropy_with_logits(output, ones)
                errD_real.backward()
                logger.add_loss("errD_real", errD_real)

                if c.train_inner_gan:
                    errD_fake = update_discriminator_fake(
                        discriminator,
                        generator_in,
                        fake_targets_onehot,
                        fake_labels,
                        (samples.size(0), c.nz, 1, 1),
                    )
                    logger.add_loss("errD_fake_in", errD_fake)

                if c.train_inner_inn:
                    errD_fake = update_discriminator_fake(
                        discriminator,
                        generator_in,
                        fake_labels,
                        fake_cond,
                        (samples.size(0), c.nch * c.img_width * c.img_width),
                        rev=True,
                    )
                    logger.add_loss("errD_fake_in", errD_fake)

                if c.train_outer_gan:
                    errD_fake = update_discriminator_fake(
                        discriminator,
                        generator_out,
                        fake_targets_onehot,
                        fake_labels,
                        (samples.size(0), c.nz, 1, 1),
                    )
                    logger.add_loss("errD_fake_out", errD_fake)

                if c.train_outer_inn:
                    pass
                    # logger.add_loss("errD_fake", errD_fake)

                optimizer_D.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar(
                        "Loss/Discriminator",
                        sum(
                            map(
                                lambda k: k[1][-1],
                                filter(
                                    lambda i: i[0].startswith("errD"),
                                    logger.losses.items(),
                                ),
                            )
                        ),
                        n_iter,
                    )

            ###
            # Update In-Dist Generator
            ###

            if c.train_inner_gan:
                optimizer_Gin.zero_grad()

                errG = update_generator_gan(
                    generator_in, discriminator, fake_targets_onehot, fake_labels, noise
                )

                optimizer_Gin.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar("Loss/In-Dist Generator", errG, n_iter)
                logger.add_loss("errG_in", errG)

            if c.train_inner_inn:
                optimizer_Gin.zero_grad()

                output = generator_in(samples, cond)
                # neg_log_likeli = torch.mean(
                #     (output - generator_in.mu[targets]
                #      / torch.exp(generator_in.log_sig[targets]))**2
                #     / 2 + generator_in.log_sig[targets], dim=1)

                mean = torch.zeros(
                    10, output.shape[1], dtype=torch.float, device=device
                )
                var = torch.zeros(10, output.shape[1], dtype=torch.float, device=device)
                for i in range(10):
                    mean[i] = output[targets == i].mean(dim=0)
                    var[i] = output[targets == i].var(dim=0)

                neg_log_likeli = torch.mean(
                    (output - mean[targets]) ** 2 / var[targets] / 2
                    + 0.5 * torch.log(var[targets]),
                    dim=1,
                )
                # zz = torch.sum(output**2, dim=1)
                # jac = generator_in.jacobian(run_forward=False)

                # neg_log_likeli = 0.5 * zz - jac

                errG = torch.mean(neg_log_likeli)
                errG.backward(retain_graph=True)

                logger.add_loss("nll_in", errG.data)

                if c.use_discriminator:
                    z = torch.randn(
                        samples.size(0),
                        c.nch * c.img_width * c.img_width,
                        device=device,
                    )

                    l_rev = update_generator_gan(
                        generator_in, discriminator, fake_cond, fake_labels, z, rev=True
                    )

                    logger.add_loss("rec_in", l_rev.data)

                else:
                    l_rev = 0

                mll = torch.tensor(0, dtype=torch.float, device=device)
                if c.use_min_likelihood:
                    try:
                        ood_samples, _ = next(ood_iter)
                    except StopIteration:
                        ood_iter = iter(data.letter_loader)
                        ood_samples, _ = next(ood_iter)

                    ood_samples = ood_samples.to(device)
                    output = generator_in(ood_samples, cond)
                    zz = torch.sum(output ** 2, dim=1)
                    jac = generator_in.jacobian(run_forward=False)

                    neg_log_likeli = 0.5 * zz - jac

                    mll = -1 * c.lambda_mll * torch.mean(neg_log_likeli)
                    mll.backward(retain_graph=True)

                    logger.add_loss("mll_in", mll.data)

                optimizer_Gin.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar(
                        "Loss/Negative Log-Likelihood In-Dist", errG, n_iter
                    )
                    writer.add_scalar("Loss/Log-Likelihood Out-Dist", mll, n_iter)
                    writer.add_scalar("Loss/Reconstruction Loss In-Dist", l_rev, n_iter)

            if c.train_outer_gan:
                optimizer_Gout.zero_grad()

                errG = update_generator_gan(
                    generator_out,
                    discriminator,
                    fake_targets_onehot,
                    fake_labels,
                    noise,
                )
                losses.append(errG.data)

                kl_loss = (
                    c.beta_generator
                    * c.ncl
                    * F.kl_div(
                        F.log_softmax(classifier(fake_samples), dim=1),
                        uniform_dist,
                        reduction="batchmean",
                    )
                )
                kl_loss.backward()
                losses.append(kl_loss.data)

                optimizer_Gout.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar("Loss/Out-Dist Generator", errG, n_iter)
                    writer.add_scalar("Loss/KL_Div", kl_loss, n_iter)

            if c.train_outer_inn:
                optimizer_Gout.zero_grad()

                output = generator_out(samples, cond)
                zz = torch.sum(output ** 2, dim=1)
                jac = generator_out.jacobian(run_forward=False)

                neg_log_likeli = 0.5 * zz - jac

                errG = torch.mean(neg_log_likeli)
                errG.backward(retain_graph=True)

                losses.append(errG.data)

                z = torch.randn(
                    samples.size(0), c.nch * c.img_width * c.img_width, device=device
                )

                fake_samples = generator_out(z, fake_cond, rev=True)
                kl_loss = (
                    c.beta_generator
                    * c.ncl
                    * F.kl_div(
                        F.log_softmax(classifier(fake_samples), dim=1),
                        uniform_dist,
                        reduction="batchmean",
                    )
                )
                kl_loss.backward()
                losses.append(kl_loss.data)

                # output = discriminator(fake_samples, fake_labels).reshape(-1)
                # l_rev = F.binary_cross_entropy_with_logits(
                #     output, ones
                # )
                # l_rev.backward()

                # l_rev = torch.tensor(0)

                # losses.append(l_rev.data)

                optimizer_Gout.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar(
                        "Loss/Negative Log-Likelihood Out-Dist", errG, n_iter
                    )
                    writer.add_scalar(
                        "Loss/Reconstruction Loss Out-Dist", kl_loss, n_iter
                    )

        logger.flush()

        if c.use_discriminator:
            scheduler_D.step()
        if c.train_inner_gan or c.train_inner_inn:
            scheduler_Gin.step()
        if c.train_outer_gan:
            scheduler_Gout.step()

        with torch.no_grad():
            if c.train_inner_gan:
                samples = generator_in(fixed_noise, onehot[fixed_targets])
                writer.add_image("Samples/In-Distribution", tensor2imgs(samples), epoch)
            if c.train_inner_inn:
                # fixed_noise_inn = torch.normal(generator_in.mu[fixed_targets],
                #                      generator_in.log_sig[fixed_targets])

                # Ideally calculate mean, var of complete dataset, use last
                # used batch as estimate TODO
                fixed_noise_inn = torch.normal(mean[fixed_targets], var[fixed_targets])
                samples = generator_in(
                    fixed_noise_inn,
                    [
                        fill[:, :, :16, :16][fixed_targets],
                        fill[:, :, :8, :8][fixed_targets],
                        make_cond(fixed_targets),
                    ],
                    rev=True,
                )
                writer.add_image("Samples/In-Distribution", tensor2imgs(samples), epoch)
            if c.train_outer_gan:
                samples = generator_out(fixed_noise, onehot[fixed_targets])
                writer.add_image(
                    "Samples/Out-of-Distribution", tensor2imgs(samples), epoch
                )
            if c.train_outer_inn:
                samples = generator_out(
                    fixed_noise_inn, make_cond(fixed_targets), rev=True
                )
                writer.add_image(
                    "Samples/Out-of-Distribution", tensor2imgs(samples), epoch
                )

        if c.use_discriminator:
            torch.save(discriminator, checkpoints_path / "discriminator.pt")
        if c.train_inner_gan or c.train_inner_inn:
            torch.save(generator_in.state_dict(), checkpoints_path / "generator_in.pt")
        if c.train_outer_gan:
            torch.save(
                generator_out.state_dict(), checkpoints_path / "generator_out.pt"
            )

        writer.flush()

except KeyboardInterrupt:
    print(red("Interrupted", bold=True))

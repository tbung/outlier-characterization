from pathlib import Path
from crayons import cyan, white, red

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms

import numpy as np

from tqdm import tqdm

import config as c
from models import Classifier, Generator, Discriminator


CHECKPOINTS_PATH = Path('./checkpoints')
CHECKPOINTS_PATH.mkdir(exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()

config_dict = {}
for k in dir(c):
    if k.startswith('_'):
        continue
    v = eval(f'c.{k}')
    config_dict[k] = v

writer.add_hparams(config_dict, {})


def tensor2imgs(t):
    imgrid = torchvision.utils.make_grid(t, c.ncl)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def pretrain_classifier(restore=True):
    print(cyan("Pretraining Classifier", bold=True))
    if restore and (CHECKPOINTS_PATH / 'classifier_mnist.pth').is_file():
        return torch.load(CHECKPOINTS_PATH / 'classifier_mnist.pth')

    model = Classifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr_classifier)

    for epoch in range(c.n_epochs_pretrain):
        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader)):
            n_iter = i + epoch * len(train_loader)
            x = x.to(device)
            y = y.to(device)

            noise = torch.randn_like(x).to(device)

            uniform_dist = torch.Tensor(x.size(0),
                                        c.ncl).fill_((1/c.ncl)).to(device)

            optimizer.zero_grad()

            output = model(x)
            loss = F.cross_entropy(output, y)
            kl_loss = F.kl_div(
                F.log_softmax(model(noise), dim=1), uniform_dist,
                reduction='batchmean'
            )

            if n_iter % c.log_interval == 0:
                writer.add_scalar("Classifier/Cross Entropy", loss, n_iter)
                writer.add_scalar("Classifier/KL Div", kl_loss, n_iter)

            loss += c.beta_classifier * c.ncl * kl_loss
            loss.backward()

            optimizer.step()

        model.eval()
        test_error = 0
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            output = torch.argmax(F.softmax(model(x), dim=1), dim=1)
            test_error += (output != y).sum()/y.size(0)

        print(f"Test Error: {test_error/len(test_loader):.5f}")
        noise_output = F.softmax(model(noise), dim=1)[0].data.cpu()
        print(f"Noise Output: {noise_output}")

    torch.save(model, CHECKPOINTS_PATH / 'classifier_mnist.pth')

    return model


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Pad(2),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5,), (0.5,)),
                       transforms.Lambda(lambda x: x.repeat(c.nch, 1, 1)),
                   ])),
    batch_size=c.batch_size, shuffle=True, pin_memory=True, num_workers=0,
    drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.Pad(2),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       transforms.Lambda(lambda x: x.repeat(c.nch, 1, 1)),
                   ])),
    batch_size=c.batch_size, shuffle=True, pin_memory=True, num_workers=0,
    drop_last=True
)

if c.train_inner_gan:
    generator_in = Generator(conditional=c.conditional).to(device)
    generator_in.apply(weights_init)
    optimizer_Gin = torch.optim.Adam(generator_in.parameters(),
                                     lr=c.lr_generator,
                                     betas=[0.5, 0.999])
    scheduler_Gin = torch.optim.lr_scheduler.StepLR(optimizer_Gin, c.lr_step,
                                                    c.lr_decay)
if c.train_outer_gan:
    generator_out = Generator(conditional=c.conditional).to(device)
    generator_out.apply(weights_init)
    optimizer_Gout = torch.optim.Adam(generator_out.parameters(),
                                      lr=c.lr_generator,
                                      betas=[0.5, 0.999])
    scheduler_Gout = torch.optim.lr_scheduler.StepLR(optimizer_Gout, c.lr_step,
                                                     c.lr_decay)
    if c.pretrain_classifier:
        classifier = pretrain_classifier(c.restore)
    else:
        classifier = Classifier().to(device)

discriminator = Discriminator(conditional=c.conditional).to(device)
discriminator.apply(weights_init)

optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=c.lr_discriminator,
                               betas=[0.5, 0.999])
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, c.lr_step,
                                              c.lr_decay)

# Fixed noise to create test samples from
fixed_noise = torch.randn(100, c.nz, 1, 1, device=device)
fixed_targets = torch.tensor(list(range(c.ncl)),
                             device=device).repeat_interleave(c.ncl)
onehot = torch.zeros(c.ncl, c.ncl)
onehot = onehot.scatter_(1,
                         torch.LongTensor(list(range(c.ncl))).view(c.ncl,
                                                                         1),
                         1).view(c.ncl, c.ncl, 1, 1)
onehot = onehot.to(device)
fill = torch.zeros((c.ncl, c.ncl, 32, 32), device=device)
for i in range(c.ncl):
    fill[i, i, :, :] = 1

ones = torch.ones((c.batch_size,), device=device).float()
zeros = torch.zeros((c.batch_size,), device=device).float()
uniform_dist = torch.Tensor(c.batch_size, c.ncl).fill_((1/c.ncl)).to(device)


print(cyan("Begin GAN Training", bold=True))
try:
    headers = ["Epoch", "errD_real"]
    if c.train_inner_gan:
        headers.extend(["errD_fake_in", "errG_in"])
    if c.train_outer_gan:
        headers.extend(["errD_fake_out", "errG_out", "errG_kl"])
    if c.train_inner_gan and c.train_outer_gan:
        headers[3], headers[4] = headers[4], headers[3]

    loss_format = "{:>15}" * len(headers)
    print(white(loss_format.format(*headers), bold=True))

    for epoch in range(c.n_epochs):

        loss_history = []

        for n, (samples, targets) in enumerate(tqdm(train_loader,
                                                    leave=False,
                                                    mininterval=1.)):
            losses = []
            n_iter = n + (epoch * len(train_loader))

            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            labels = fill[targets]

            noise = torch.randn((samples.size(0), c.nz, 1, 1), device=device)
            fake_targets = torch.randint(c.ncl, (samples.size(0),),
                                         device=device)

            ###
            # Update Discriminator
            ###

            optimizer_D.zero_grad()

            output = discriminator(samples, labels).reshape(-1)
            errD_real = F.binary_cross_entropy_with_logits(
                output, ones
            )
            errD_real.backward()
            losses.append(errD_real)

            if c.train_inner_gan:
                fake_samples = generator_in(noise, onehot[fake_targets])
                fake_labels = fill[fake_targets]
                output = discriminator(fake_samples.detach(), fake_labels).reshape(-1)
                errD_fake = F.binary_cross_entropy_with_logits(
                    output, zeros
                )
                errD_fake.backward()
                losses.append(errD_fake)

            if c.train_outer_gan:
                fake_samples = generator_out(noise, onehot[fake_targets])
                fake_labels = fill[fake_targets]
                output = discriminator(fake_samples.detach(), fake_labels).reshape(-1)
                errD_fake = F.binary_cross_entropy_with_logits(
                    output, zeros
                )
                errD_fake.backward()
                losses.append(errD_fake)

            optimizer_D.step()

            if n_iter % c.log_interval == 0:
                writer.add_scalar('Loss/Discriminator', sum(losses), n_iter)

            ###
            # Update In-Dist Generator
            ###

            if c.train_inner_gan:
                optimizer_Gin.zero_grad()

                fake_samples = generator_in(noise, onehot[fake_targets])
                fake_labels = fill[fake_targets]
                output = discriminator(fake_samples, fake_labels).reshape(-1)
                errG = F.binary_cross_entropy_with_logits(
                    output, ones
                )
                errG.backward()

                optimizer_Gin.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar('Loss/In-Dist Generator', errG, n_iter)
                losses.append(errG)

            if c.train_outer_gan:
                optimizer_Gout.zero_grad()

                fake_samples = generator_out(noise, onehot[fake_targets])
                fake_labels = fill[fake_targets]
                output = discriminator(fake_samples, fake_labels).reshape(-1)
                errG = F.binary_cross_entropy_with_logits(
                    output, ones
                )
                errG.backward(retain_graph=True)
                losses.append(errG.data)

                kl_loss = c.beta_generator * c.ncl * F.kl_div(
                    F.log_softmax(classifier(fake_samples), dim=1),
                    uniform_dist, reduction='batchmean'
                )
                kl_loss.backward()
                losses.append(kl_loss.data)

                optimizer_Gout.step()

                if n_iter % c.log_interval == 0:
                    writer.add_scalar('Loss/Out-Dist Generator', errG, n_iter)
                    writer.add_scalar('Loss/KL_Div', kl_loss, n_iter)

            loss_history.append(losses)

        print(loss_format.format(f"{epoch}",
                                 *[f"{l:.3f}" for l in np.mean(np.array(loss_history), axis=0)]))

        scheduler_D.step()
        if c.train_inner_gan:
            scheduler_Gin.step()
        if c.train_outer_gan:
            scheduler_Gout.step()

        with torch.no_grad():
            if c.train_inner_gan:
                samples = generator_in(fixed_noise, onehot[fixed_targets])
                writer.add_image(
                    'Samples/In-Distribution',
                    tensor2imgs(samples),
                    epoch
                )
            if c.train_outer_gan:
                samples = generator_out(fixed_noise, onehot[fixed_targets])
                writer.add_image(
                    'Samples/Out-of-Distribution',
                    tensor2imgs(samples),
                    epoch
                )

except KeyboardInterrupt:
    # TODO: Checkpointing
    print(red("Interrupted", bold=True))

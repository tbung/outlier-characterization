from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import data
from models import Discriminator, INN_AA
from utils import logger, config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Don't need epochs as we can generate infinitely many samples
max_iter = int(5e5)
batch_size = 512

logger = logger.LossLogger()
writer = SummaryWriter(comment='_DISCRIMINATOR')

checkpoints_path = Path(writer.get_logdir()) / 'checkpoints'
checkpoints_path.mkdir(exist_ok=True)

c = config.Config()
c.load('../mnist/config/default.toml')


def make_cond(labels):
    cond_tensor = torch.zeros(labels.size(0), 10).cuda()
    cond_tensor.scatter_(1, labels.view(-1, 1), 1.)
    return cond_tensor


def b(log_n):
    if type(log_n) is not torch.Tensor:
        log_n = torch.tensor(log_n, device=device, dtype=torch.float)

    return torch.sqrt(2 * log_n) - (torch.log(log_n) + torch.log(4 * torch.tensor(np.pi)))/(2 * torch.sqrt(2 * log_n))


def a(log_n):
    if type(log_n) is not torch.Tensor:
        log_n = torch.tensor(log_n, device=device, dtype=torch.float)

    return torch.sqrt(2 * log_n)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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


inn = INN_AA(9, interpolation='linear',
             weight_norm_exp=2, weight_norm_constraint=0.9).to(device)
# inn.load_state_dict(torch.load('runs/Feb06_14-23-13_GLaDOS/checkpoints/generator_in.pt'))
inn.load_state_dict(dict(filter(lambda x: 'tmp' not in x[0],
                                torch.load('runs/Jun28_16-14-27_GLaDOS_INN/checkpoints/innaa.pt').items())))

model = Discriminator(c, conditional=True).to(device)
model.apply(weights_init)
optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-4, betas=[0.5, 0.999]
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1e-2**(1/1e3))

fill = torch.zeros((10, 10, 32, 32), device=device)
for i in range(10):
    fill[i, i, :, :] = 1

ones = torch.ones((batch_size,), device=device).float()
zeros = torch.zeros((batch_size,), device=device).float()

z_fixed = build_z_fixed(9)

for i in tqdm(range(max_iter)):

    with torch.no_grad():
        A = torch.randn(512, 10, device=device)
        A_exp = torch.exp(A)
        A = A_exp/(A_exp**2).sum(dim=1, keepdim=True)**(1/2)
        A = 0.4 * A

        targets = torch.randint(10, (batch_size,), device=device)
        cond = [
            fill[:, :, :16, :16][targets],
            fill[:, :, :8, :8][targets],
            make_cond(targets)
        ]

        samples, _ = inn.sample(A, z_fixed, cond)

        # targets = F.softmax(targets, dim=1).max(dim=1)[1]

        A = torch.randn(512, 10, device=device)
        A[np.arange(512), torch.randint(10, (512,))] = 0
        A[A != 0] = torch.exp(A[A != 0])
        A = A/(A**2).sum(dim=1, keepdim=True)**(1/2)
        A = (0.8 - 0.4) * A + 0.4
        out_samples, _ = inn.sample(A, z_fixed, cond)
        # out_targets = F.softmax(out_targets, dim=1).max(dim=1)[1]


    optimizer.zero_grad()
    # samples = F.pad(samples, (2, 2, 2, 2))
    # out_samples = F.pad(out_samples, (2, 2, 2, 2))

    output_in = model(samples.detach(), fill[targets]).reshape(-1)
    errD_real = F.binary_cross_entropy_with_logits(
        output_in, ones
    )
    errD_real.backward()
    logger.add_loss("err_in", errD_real*1e3)
    writer.add_scalar("disc_training/bce_in", errD_real, i)

    output_out = model(out_samples.detach(), fill[targets]).reshape(-1)
    errD_fake = F.binary_cross_entropy_with_logits(
        output_out, zeros
    )
    errD_fake.backward()
    logger.add_loss("err_out", errD_fake*1e3)
    writer.add_scalar("disc_training/bce_out", errD_fake, i)

    optimizer.step()

    if i % 500 == 0:
        logger.flush()
        scheduler.step()

        with torch.no_grad():
            # print(f"Output IN: {F.sigmoid(output_in).mean().item()}")
            # print(f"Output OUT: {F.sigmoid(output_out).mean().item()}")
            writer.add_scalar("disc_training/sample_in", torch.sigmoid(output_in).mean().item(), i)
            writer.add_scalar("disc_training/sample_out", torch.sigmoid(output_out).mean().item(), i)
            it = iter(data.train_loader)
            x, y = next(it)
            x = x.to(device)
            y = y.to(device)
            # x = F.pad(x, (2, 2, 2, 2))
            output = model(x, fill[y]).reshape(-1)
            # print(f"Train set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("disc_training/sample_train", torch.sigmoid(output).mean().item(), i)

            it = iter(data.test_loader)
            x, y = next(it)
            x = x.to(device)
            y = y.to(device)
            # x = F.pad(x, (2, 2, 2, 2))
            output = model(x, fill[y]).reshape(-1)
            # print(f"Test set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("disc_training/sample_test", torch.sigmoid(output).mean().item(), i)

            it = iter(data.letter_loader)
            x, _ = next(it)
            x = x.to(device)
            # x = F.pad(x, (2, 2, 2, 2))
            # reuse y of test set because letter classes make no sense here
            # (and will crash)
            output = model(x, fill[y]).reshape(-1)
            # print(f"Letter set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("disc_training/sample_letter", torch.sigmoid(output).mean().item(), i)
            writer.flush()

        torch.save(model.state_dict(), checkpoints_path / 'discriminator.pt')

# check on train set sample
# check on test set sample
# check on letters

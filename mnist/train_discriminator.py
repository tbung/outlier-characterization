from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import data
from models import INN, Discriminator
from logger import LossLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Don't need epochs as we can generate infinitely many samples
max_iter = int(5e5)
batch_size = 512

logger = LossLogger()
writer = SummaryWriter(comment='_DISCRIMINATOR')

checkpoints_path = Path(writer.get_logdir()) / 'checkpoints'
checkpoints_path.mkdir(exist_ok=True)


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


inn = INN().to(device)
# inn.load_state_dict(torch.load('runs/Feb06_14-23-13_GLaDOS/checkpoints/generator_in.pt'))
inn.load_state_dict(dict(filter(lambda x: 'tmp' not in x[0], torch.load('runs/May13_19-39-26_GLaDOS/checkpoints/generator_in.pt').items())))

model = Discriminator(conditional=True).to(device)
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

# Initialize sampling distribution
latent = torch.empty(0, 32 * 32)
classes = torch.empty(0).long()
with torch.no_grad():
    for x, y in tqdm(data.train_loader):
        x = x.to(device)
        y = y.to(device)
        cond = [
            fill[:, :, :16, :16][y],
            fill[:, :, :8, :8][y],
            make_cond(y)
        ]
        output = inn(x, cond)
        latent = torch.cat([latent, output.data.cpu()])
        classes = torch.cat([classes, y.data.cpu()])

mean = latent.mean(dim=0).to(device)
cov = torch.tensor(np.cov(latent.cpu().numpy().T), device=device,
                   dtype=torch.float)
w, v = torch.eig(cov, eigenvectors=True)
s = 1.5
# latent_dist = torch.distributions.multivariate_normal.MultivariateNormal(m, 9*cov)
std_normal = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(32*32, device=device),
    torch.eye(32*32, device=device)
)
gumbel = torch.distributions.gumbel.Gumbel(32+b((s*32 - 32)**2)*0.7, 1/a((s*32
                                                                          - 32)**2)*0.7)
typical = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)


for i in tqdm(range(max_iter)):

    with torch.no_grad():
        targets = torch.randint(10, (batch_size,),
                                device=device)
        # cond = make_cond(targets)
        cond = [
            fill[:, :, :16, :16][targets],
            fill[:, :, :8, :8][targets],
            make_cond(targets)
        ]
        labels = fill[targets]

        # IN SAMPLE
        samples = inn(typical.sample((batch_size,)), cond, rev=True)

        # OUT SAMPLE
        z = std_normal.sample((batch_size,))
        z /= torch.norm(z, dim=1)[:, None]

        # sample mahalanobis radius and convert to euclidean
        r = gumbel.sample((batch_size,)).to(device)
        r = r[:, None, None] * (torch.sqrt(torch.diag(w[:, 0])) @ v.T)[None, :, :]

        # combine direction and radius
        z = z[:, None, :] @ r
        z = z.reshape(-1, 32*32)
        z += mean
        out_samples = inn(z, cond, rev=True)

    optimizer.zero_grad()
    # samples = F.pad(samples, (2, 2, 2, 2))

    output_in = model(samples.detach(), labels).reshape(-1)
    errD_real = F.binary_cross_entropy_with_logits(
        output_in, ones
    )
    errD_real.backward()
    logger.add_loss("err_in", errD_real*1e3)
    writer.add_scalar("disc_training/bce_in", errD_real, i)

    output_out = model(out_samples.detach(), labels).reshape(-1)
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
            output = model(x, fill[y]).reshape(-1)
            # print(f"Train set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("disc_training/sample_train", torch.sigmoid(output).mean().item(), i)

            it = iter(data.test_loader)
            x, y = next(it)
            x = x.to(device)
            y = y.to(device)
            output = model(x, fill[y]).reshape(-1)
            # print(f"Test set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("disc_training/sample_test", torch.sigmoid(output).mean().item(), i)

            it = iter(data.letter_loader)
            x, _ = next(it)
            x = x.to(device)
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

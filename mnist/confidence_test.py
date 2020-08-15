# Train classifier and discriminator using pretrained INN

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision

from models import INN, Classifier
from logger import LossLogger
from utils import config
import data

from tqdm import tqdm
from crayons import cyan
from pathlib import Path

import matplotlib.pyplot as plt

c = config.Config()
c.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Don't need epochs as we can generate infinitely many samples
# max_iter = int(5e5)
max_iter = 1001
batch_size = 512

logger = LossLogger()
writer = SummaryWriter(comment='_CLASSIFIER')

checkpoints_path = Path(writer.get_logdir()) / 'checkpoints'
checkpoints_path.mkdir(exist_ok=True)


def make_cond(labels):
    cond_tensor = torch.zeros(labels.size(0), c.ncl).cuda()
    if c.conditional:
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.)
    else:
        cond_tensor[:, 0] = 1
    return cond_tensor


def b(log_n):
    if type(log_n) is not torch.Tensor:
        log_n = torch.tensor(log_n, device=device, dtype=torch.float)

    return torch.sqrt(2 * log_n) - (torch.log(log_n) + torch.log(4 * torch.tensor(np.pi)))/(2 * torch.sqrt(2 * log_n))


def a(log_n):
    if type(log_n) is not torch.Tensor:
        log_n = torch.tensor(log_n, device=device, dtype=torch.float)

    return torch.sqrt(2 * log_n)


inn = INN().to(device)
inn.load_state_dict(dict(filter(lambda x: 'tmp' not in x[0], torch.load('runs/May13_19-39-26_GLaDOS/checkpoints/generator_in.pt').items())))
# inn.load_state_dict(torch.load('runs/Jan21_00-34-42_GLaDOS/checkpoints/generator_in.pt'))

print(cyan("Training Classifier", bold=True))

# model = Classifier().to(device)
model = torchvision.models.vgg11(num_classes=10).to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1e-2**(1e-2))

# with torch.no_grad():
#     correct = 0
#     for x, y in data.test_loader:
#         x, y = x.to(device), y.to(device)
#         x = x.repeat((1, 3, 1, 1))
#         y_ = model(x).max(dim=1)[1]
#         print(y)
#         print(y_)
#         correct += (y_ == y).sum()
#         print(correct)
#         break

fill = torch.zeros((10, 10, 32, 32), device=device)
for i in range(10):
    fill[i, i, :, :] = 1

# latent = torch.empty(0, 32 * 32)
# classes = torch.empty(0).long()
# with torch.no_grad():
#     for x, y in tqdm(data.train_loader):
#         x = x.to(device)
#         y = y.to(device)
#         cond = [
#             fill[:, :, :16, :16][y],
#             fill[:, :, :8, :8][y],
#             make_cond(y)
#         ]
#         output = inn(x, cond)
#         latent = torch.cat([latent, output.data.cpu()])
#         classes = torch.cat([classes, y.data.cpu()])

# mean = latent.mean(dim=0).to(device)
# cov = torch.tensor(np.cov(latent.cpu().numpy().T), device=device,
#                    dtype=torch.float)
# torch.save(mean, './mean.pt')
# torch.save(cov, './cov.pt')
mean = torch.load('./mean.pt')
cov = torch.load('./cov.pt')
w, v = torch.eig(cov, eigenvectors=True)
s = 1.5
# latent_dist = torch.distributions.multivariate_normal.MultivariateNormal(m, 9*cov)
std_normal = torch.distributions.multivariate_normal.MultivariateNormal(
    torch.zeros(32*32, device=device),
    torch.eye(32*32, device=device)
)
gumbel = torch.distributions.gumbel.Gumbel(32+b((s*32 - 32)**2)*0.7, 1/a((s*32 - 32)**2)*0.7)
typical = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
uniform_dist = torch.Tensor(512, c.ncl).fill_((1/c.ncl)).to(device)

# del latent
# del classes

train_iter = iter(data.train_loader)

# torch.save(model, CHECKPOINTS_PATH / 'classifier_mnist.pth')
for i in tqdm(range(max_iter)):
    # try:
    #     samples, targets = next(train_iter)
    # except:
    #     train_iter = iter(data.train_loader)
    #     samples, targets = next(train_iter)

    # samples, targets = samples.to(device), targets.to(device)

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
        samples = inn(std_normal.sample((batch_size,)), cond, rev=True)

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

    samples += c.add_image_noise * torch.randn_like(samples, device=device)
    samples = samples.repeat((1, 3, 1, 1))
    samples.requires_grad = True
    out_samples += c.add_image_noise * torch.randn_like(out_samples, device=device)
    out_samples = out_samples.repeat((1, 3, 1, 1))
    out_samples.requires_grad = True

    optimizer.zero_grad()

    # print(targets)

    # plt.figure()
    # plt.imshow(torchvision.utils.make_grid(samples).data.mul(255).clamp(0,
    #                                                                     255).byte().cpu().numpy().transpose(1,
    #                                                                                                         2,
    #                                                                                                         0))
    # plt.show()

    # break

    output_in = model(samples)
    # print(F.softmax(output_in, dim=1).max(dim=1)[1] == targets)
    logger.add_loss('train_acc', (F.softmax(output_in, dim=1).max(dim=1)[1] == targets).float().mean())
    loss = F.cross_entropy(output_in, targets)
    logger.add_loss("err_in", loss)
    writer.add_scalar("class_training/cross_entropy", loss, i)

    output_out = model(out_samples.detach())

    kl_loss = F.kl_div(
        F.log_softmax(output_out, dim=1), uniform_dist,
        reduction='batchmean'
    )
    logger.add_loss("err_out", kl_loss)
    writer.add_scalar("class_training/kl_loss", kl_loss, i)

    loss += 0.1 / np.sqrt(i+1) * c.ncl * kl_loss
    loss.backward()
    # print(samples.grad)
    # print(output_in.grad)
    # print(output_in.grad_fn)
    # print(loss.grad_fn)
    # print(output_in.requires_grad)
    # break

    optimizer.step()

    if i % 500 == 0:
        scheduler.step()
        model.eval()

        with torch.no_grad():
            # print(f"Output IN: {F.sigmoid(output_in).mean().item()}")
            # print(f"Output OUT: {F.sigmoid(output_out).mean().item()}")
            writer.add_scalar("disc_training/sample_in", F.softmax(output_in, dim=1).max(dim=1)[0].mean().item(), i)
            # writer.add_scalar("disc_training/sample_out", F.softmax(output_out, dim=1).max(dim=1)[0].mean().item(), i)
            it = iter(data.train_loader)
            x, y = next(it)
            x = x.to(device)
            x = x.repeat((1, 3, 1, 1))
            y = y.to(device)
            output = model(x)
            confidence = F.softmax(output, dim=1).max(dim=1)[0].mean().item()
            # print(f"Train set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("class_training/sample_train", confidence, i)

            it = iter(data.test_loader)
            x, y = next(it)
            x = x.to(device)
            x = x.repeat((1, 3, 1, 1))
            y = y.to(device)
            output = model(x)
            confidence = F.softmax(output, dim=1).max(dim=1)[0].mean().item()
            # print(f"Train set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("class_training/sample_test", confidence, i)

            it = iter(data.letter_loader)
            x, _ = next(it)
            x = x.to(device)
            x = x.repeat((1, 3, 1, 1))
            # reuse y of test set because letter classes make no sense here
            # (and will crash)
            output = model(x)
            confidence = F.softmax(output, dim=1).max(dim=1)[0].mean().item()
            # print(f"Train set output: {F.sigmoid(output).mean().item()}")
            writer.add_scalar("class_training/sample_letter", confidence, i)

            for x, y in data.test_loader:
                x, y = x.to(device), y.to(device)
                x = x.repeat((1, 3, 1, 1))
                y_ = F.softmax(model(x), dim=1).max(dim=1)[1]
                logger.add_loss("test_acc", (y_ == y).float().mean())

            # print(y)
            # print(y_)
            # print(correct)

            # writer.add_scalar("class_training/accuracy_test", acc, i)

        writer.flush()
        logger.flush()

        model.train()
        torch.save(model.state_dict(), checkpoints_path / 'classifier.pt')

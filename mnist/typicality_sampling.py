import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

import config as c
from models import INN

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def tensor2imgs(t):
    imgrid = torchvision.utils.make_grid(t, c.ncl)
    return imgrid.data.mul(255).clamp(0, 255).byte().cpu().numpy()


def make_cond(labels):
    cond_tensor = torch.zeros(labels.size(0), c.ncl).cuda()
    if c.conditional:
        cond_tensor.scatter_(1, labels.view(-1, 1), 1.)
    else:
        cond_tensor[:, 0] = 1
    return cond_tensor


fixed_targets = torch.tensor(list(range(c.ncl)),
                             device=device).repeat_interleave(c.ncl)

model = INN().to(device)
model.load_state_dict(torch.load('runs/Jan14_16-38-26_GLaDOS/checkpoints/generator_in.pt'))
# print(model)

classifier = torch.load('checkpoints/classifier_mnist.pth')


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

out_pos = torch.empty(0, 32 * 32, device=device)
classes = torch.empty(0, device=device).long()
d_train = torch.empty(0, device=device)
with torch.no_grad():
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)
        output = model(x, make_cond(y))
        d_train = torch.cat([d_train, torch.norm(output, dim=1)])
        out_pos = torch.cat([out_pos, output])
        classes = torch.cat([classes, y])

# d_test = torch.empty(0, device=device)
# with torch.no_grad():
#     for x, y in tqdm(test_loader):
#         x = x.to(device)
#         y = y.to(device)
#         output = model(x, make_cond(y))
#         d_test = torch.cat([d_test, torch.norm(output, dim=1)])

# for i in range(10):
#     plt.hist(d_train[classes == i].cpu().numpy(), bins=100, density=True,
#              alpha=0.2)
#     # plt.axvline(d_train.mean(), c='k')
#     # plt.axvline(d_train.mean() + 3*d_train.std(), c='r')
#     # plt.axvline(d_train.mean() - 3*d_train.std(), c='r')

#     print(d_train[classes == i].mean())
#     print(d_train[classes == i].std())

# plt.show()


# print(d_test.min())
# print(d_test.max())
# print(d_test.mean())
# print(d_test.median())
# print((d_test > 1e13).shape)
# d_test_hist, edges = np.histogram(d_test[d_test < 1e3].cpu().numpy(), bins=100)
# print(edges[np.argmax(d_test_hist)])
# plt.hist(d_test[d_test < 1e3].cpu().numpy(), bins=100, density=True)
# plt.show()

# extreme_noise_inn = torch.empty(0, c.nch * 32 * 32, device=device)

# while extreme_noise_inn.size(0) < 100:
#     x = torch.randn(10000, c.nch * 32 * 32, device=device)
#     extreme_noise_inn = torch.cat([
#         extreme_noise_inn,
#         x[torch.tensor(np.logical_and(
#             (torch.norm(x, dim=1) > d_train.mean() + 3 * d_train.std()).cpu(),
#             (torch.norm(x, dim=1) < d_train.mean() + 6 * d_train.std()).cpu()
#         )).bool()]
#     ])
#     extreme_noise_inn = torch.cat([
#         extreme_noise_inn,
#         x[torch.tensor(np.logical_and(
#             (torch.norm(x, dim=1) < d_train.mean() - 3 * d_train.std()).cpu(),
#             (torch.norm(x, dim=1) > d_train.mean() - 6 * d_train.std()).cpu()
#         )).bool()]
#     ])
#     print(extreme_noise_inn.shape)
model.eval()
classifier.eval()

for i in range(10):
    m = out_pos[classes == i].mean(dim=0).cpu().numpy()
    cov = np.cov(out_pos[classes == i].cpu().numpy().T)
    typical_noise = torch.tensor(np.random.multivariate_normal(m, cov, 100),
                                 device=device).float()

    extreme_noise = np.random.multivariate_normal(m, 9*cov, 5000)
    d_temp = np.sqrt((extreme_noise - m)[:, None, :] @ np.linalg.inv(cov) @
                     (extreme_noise - m)[:, :, None]).reshape(-1)
    print((d_temp > 3*32).sum())
    print((d_temp < 7).sum())
    extreme_noise = torch.tensor(extreme_noise[np.logical_or(d_temp > 3*32,
                                                             d_temp < 7)],
                                 device=device).float()
    print(extreme_noise.shape)

    samples = model(typical_noise, make_cond(i * torch.ones(100, device=device).long()), rev=True)
    confidence = torch.nn.functional.softmax(classifier(samples), dim=1)
    print(f"ID confidence max: {confidence.max(dim=1)[0].mean().item()}")
    plt.imshow(tensor2imgs(samples.reshape(-1, 1, 32, 32)).transpose(1, 2, 0))
    plt.show()

    ts_sample, ts_target = next(iter(train_loader))
    ts_sample = ts_sample.to(device)
    ts_target = ts_target.to(device)
    print(f"Generated sample statistic: {samples.mean().item():.2f} +/- {samples.std().item():.2f}")
    print(f"GT sample statistic: {ts_sample.mean().item():.2f} +/- {ts_sample.std().item():.2f}")
    recreated = model(model(ts_sample, make_cond(ts_target)), make_cond(ts_target), rev=True)
    print(f"MSE: {((ts_sample - recreated)**2).sum(dim=1).mean().item()}")
    print(f"Recreated confidence: {torch.nn.functional.softmax(classifier(recreated), dim=1).max(dim=1)[0].mean().item()}")
    print(f"GT confidence: {torch.nn.functional.softmax(classifier(ts_sample), dim=1).max(dim=1)[0].mean().item()}")

    samples = model(extreme_noise[np.random.permutation(extreme_noise.size(0))][:100], make_cond(i * torch.ones(100, device=device).long()), rev=True)

    confidence = torch.nn.functional.softmax(classifier(samples), dim=1)
    print(f"OOD confidence max: {confidence.max(dim=1)[0].mean().item()}")
    plt.imshow(tensor2imgs(samples.reshape(-1, 1, 32, 32)).transpose(1, 2, 0))
    plt.show()

# Experiment
train_inner_gan = False
train_inner_inn = False

assert (not (train_inner_inn and train_inner_gan)), "In-Distribution network can either be a DCGAN or an INN"

train_outer_gan = True
pretrain_classifier = True
restore = True
conditional = True

# DCGAN Parameters
nch = 3
nz = 100
ngf = 64
ndf = 64
ncl = 10

# Training Parameters
n_epochs = 50
n_epochs_pretrain = 50

batch_size = 128
lr_generator = 2e-4
lr_discriminator = 2e-4
lr_classifier = 1e-3
lr_step = 30
lr_decay = 1

beta_classifier = 1
beta_generator = 1

log_interval = 50

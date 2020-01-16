# Experiment
train_inner_gan = False
train_inner_inn = True

assert (not (train_inner_inn and train_inner_gan)), "In-Distribution network can either be a DCGAN or an INN"

train_outer_gan = True
train_outer_inn = False
pretrain_classifier = True
restore = False
conditional = True

# DCGAN Parameters
nch = 1
nz = 100
ngf = 64
ndf = 64
ncl = 10

# INN Parameters
n_blocks = 24
internal_width = 512
clamping = 1.5
fc_dropout = 0.0
init_scale = 0.03
add_image_noise = 0.15
eps = 1

# Training Parameters
n_epochs = 500
n_epochs_pretrain = 10

batch_size = 512
lr_generator = 1e-4
lr_discriminator = 1e-4
lr_classifier = 1e-3
lr_step = 1
lr_decay = 0.01**(1/n_epochs)

beta_classifier = 1
beta_generator = 50

log_interval = 50

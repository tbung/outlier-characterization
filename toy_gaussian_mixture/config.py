train_inner_gan = True
train_inner_inn = False

assert not (
    train_inner_inn and train_inner_gan
), "In-Distribution network can either be a GAN or an INN"

train_outer_gan = True
train_outer_inn = False

assert not (
    train_outer_inn and train_outer_gan
), "Out-of-Distribution network can either be a GAN or an INN"

pretrain_classifier = True

# GAN Parameters
nz = 100
ngf = 500
ndf = 500
ncf = 500

# Training Parameters
n_epochs = 300
n_epochs_pretrain = 100

batch_size = 400
lr_generator = 5e-4
lr_discriminator = 5e-4
lr_classifier = 1e-3
lr_schedule = 20
lr_step = 0.1

beta_classifier = 10
beta_generator = 10

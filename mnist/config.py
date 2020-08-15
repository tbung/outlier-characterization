import toml


class Config:
    def __init__(self):

        # Experiment
        self.train_inner_gan = False
        self.train_inner_inn = True

        self.train_outer_gan = False
        self.train_outer_inn = False

        self.use_discriminator = False
        self.pretrain_classifier = False
        self.restore = True

        self.conditional = True
        self.dataset = 'EMNIST'

        # DCGAN Parameters
        self.nch = 1
        self.img_width = 2*28
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.ncl = 10

        # INN Parameters
        self.couplig_type = 'GLOW'
        self.n_blocks = 24
        self.internal_width = 512
        self.clamping = 1.5
        self.fc_dropout = 0.0
        self.init_scale = 0.03
        self.add_image_noise = 0.15
        self.eps = 1

        # Training Parameters
        self.n_epochs = 1500
        self.n_epochs_pretrain = 10

        self.batch_size = 512
        self.lr_generator = 1e-4
        self.lr_discriminator = 1e-4
        self.lr_classifier = 1e-3
        self.lr_step = 1
        self.lr_decay = 0.01**(1/self.n_epochs)

        self.beta_classifier = 1
        self.beta_generator = 50

        self.log_interval = 50

    def load(self, path):
        self.__dict__.update(toml.load(path))
        self.validate()

    def validate(self):
        assert (not (self.train_inner_inn and self.train_inner_gan)), \
            "In-Distribution network can either be a DCGAN or an INN"

        assert self.dataset in ['MNIST', 'EMNIST', 'letters', 'fake'], \
            "Dataset Name not understood"

    def __str__(self):
        return str(self.__dict__)

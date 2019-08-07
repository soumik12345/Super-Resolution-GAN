from src.networks import Generator, Discriminator
from src.loss import VGGLoss


class SRGAN(object):

    def __init__(self, image_shape, downsample_factor):
        self.image_shape = image_shape
        self.downsample_factor = downsample_factor
        self.latent_dimension = (
            image_shape[0] // self.downsample_factor,
            image_shape[1] // self.downsample_factor,
            image_shape[2],
        )
        self.generator = Generator(self.latent_dimension).generator
        self.discriminator = Discriminator(self.image_shape).discriminator
        self.vgg_loss = VGGLoss(self.image_shape)
        self.build_gan()
    
    
    def build_gan(self):
        self.discriminator.trainable = False

        input_placeholder = Input(shape = self.latent_dimension)
        x = self.generator(input_placeholder)
        output = self.discriminator(x)
        gan = Model(input_placeholder, output)
        gan.compile(
            loss = [
                self.vgg_loss.loss
            ],
            loss_weights = [1.],
            optimizer = Adam(
                lr = 1E-4,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-08
            )
        )
        self.gan = gan
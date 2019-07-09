from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from generator import Generator
from discriminator import Discriminator
from vgg_loss import  VGGLoss


class SRGAN:


    def __init__(self, image_dimension, downsample_factor):
        self.image_dimension = image_dimension
        self.downsample_factor = downsample_factor
        self.latent_dimension = (
            image_dimension[0] // self.downsample_factor,
            image_dimension[1] // self.downsample_factor,
            image_dimension[2],
        )
        self.generator = Generator(self.latent_dimension)
        self.discriminator = Discriminator(self.image_dimension)
        self.vgg_loss = VGGLoss(self.image_dimension)
    

    def build_gan(self, generator, discriminator):
        """
        Build the SRGAN model

        ...

        Parameters
        ----------
        generator : tf.keras Model
            Generator Model
        discriminator : tf.keras Model
            Discriminator Model
        
        Returns
        -------
        Combined and Compiled GAN model
        """
        discriminator.trainable = False
        
        input_placeholder = Input(shape = self.latent_dimension)
        x = self.generator(input_placeholder)
        output = discriminator(x)
        
        gan = Model(input_placeholder, output)
        
        gan.compile(
            loss = [
                vgg_loss.loss,
                'binary_crossentropy'
            ], loss_weights = [1., 1e-3],
            optimizer = Adam(
                lr = 1E-4,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon = 1e-08
            )
        )
        return gan
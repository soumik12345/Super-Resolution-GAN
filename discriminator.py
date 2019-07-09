from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU
from tensorflow.keras.models import Model


class Discriminator:
    
    
    def __init__(self, image_dimension):
        self.image_dimension = image_dimension
        self.build_model()
    

    def discriminator_block(self, x, kernel_size, filters, strides):
        """
        Implementation of a Convolutional Discriminator Block
        
        ...
        
        Parameters
        ----------
        x : tensor
            The input tensor to the residual block
        kernel_size : integer
            Size of the convolutional kernels
        filters : integer
            Number of convolutional filters
        strides : integer
            Amount of convolutional strides
        
        Returns
        -------
        Output Tensor to the residual block
        """

        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same"
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        return x
    

    def build_model(self):
        """
        Build the Discriminator Model
        """

        input_placeholder = Input(shape = self.image_dimension)

        x = self.discriminator_block(input_placeholder, 3, 64, 2)
        x = self.discriminator_block(x, 3, 128, 1)
        x = self.discriminator_block(x, 3, 128, 2)
        x = self.discriminator_block(x, 3, 256, 1)
        x = self.discriminator_block(x, 3, 256, 2)
        x = self.discriminator_block(x, 3, 512, 1)
        x = self.discriminator_block(x, 3, 512, 2)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        self.model = Model(input_placeholder, x)
    

    def summary(self):
        """
        Display the model summary
        """
        self.model.summary()
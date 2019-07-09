from tensorflow.keras.layers import Dense, Flatten, Input, add
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, PReLU, Activation
from tensorflow.keras.models import Model


class Generator:

    
    def __init__(self, latent_dimension):
        self.latent_dimension = latent_dimension
        self.build_model()
    

    def residual_block(self, x, kernel_size, filters, strides):
        """
        Implementation of a Residual Block
        
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
        
        x_shortcut = x
        
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same"
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)
        x = PReLU(
            alpha_initializer = 'zeros',
            alpha_regularizer = None,
            alpha_constraint = None,
            shared_axes = [1, 2]
        )(x)
        
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same"
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)

        output = add([x_shortcut, x])

        return output
    

    def upsampling_block(self, x, kernel_size, filters, strides):
        """
        Implementation of an Upsampling(Deconvolutional) Block
        
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
        Output Tensor to the Upsampling block
        """

        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same"
        )(x)
        x = UpSampling2D(size = 2)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        return x
    

    def build_model(self):
        """
        Build the Generator Model
        """

        input_placeholder = Input(shape = self.latent_dimension)

        x = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(input_placeholder)
        x = PReLU(
            alpha_initializer = 'zeros',
            alpha_regularizer = None,
            alpha_constraint = None,
            shared_axes = [1, 2]
        )(x)

        x_shortcut = x

        for i in range(16):
            x = self.residual_block(x, 3, 64, 1)
        
        x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(x)
        x = BatchNormalization(momentum = 0.5)(x)
        
        x = add([x_shortcut, x])

        for i in range(2):
            x = self.upsampling_block(x, 3, 256, 1)
        
        x = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(x)
        x = Activation('tanh')(x)

        self.model = Model(input_placeholder, x)
    

    def summary(self):
        """
        Display the model summary
        """
        self.model.summary()
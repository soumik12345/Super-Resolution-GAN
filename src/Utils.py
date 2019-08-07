from tensorflow.keras.layers import (
    Conv2D, add,
    BatchNormalization,
    PReLU, LeakyReLU,
    Conv2DTranspose
)


class Utils:

    @staticmethod
    def residual_block(x, filters, kernel_size, strides):
        skip = x
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = 'same'
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
            padding = 'same'
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)
        x = add([skip, x])
        return x
    

    @staticmethod
    def upsampling_block(x, filters, kernel_size, strides):
        x = Conv2DTranspose(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = 'same'
        )(x)
        x = LeakyReLU(alpha = 0.2)(x)
        return x
    

    @staticmethod
    def discriminator_block(x, filters, kernel_size, strides):
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = 'same'
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)
        x = LeakyReLU(alpha = 0.2)(x)
        return x
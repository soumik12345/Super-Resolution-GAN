from tensorflow.keras.layers import (
    Conv2D, add,
    BatchNormalization,
    PReLU, LeakyReLU,
    Conv2DTranspose,
    UpSampling2D
)


class Utils:

    @staticmethod
    def residual_block(x, filters, kernel_size, strides):
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
    

    @staticmethod
    def upsampling_block(x, filters, kernel_size, strides):
        x = Conv2D(
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = "same"
        )(x)
        x = UpSampling2D(size = 2)(x)
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
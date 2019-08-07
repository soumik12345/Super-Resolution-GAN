from src.Utils import Utils
from tensorflow.keras.layers import (
    Input, Conv2D,
    PReLU, LeakyReLU, Activation,
    BatchNormalization, add,
    Flatten, Dense
)
from tensorflow.keras.models import Model


class Generator(object):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.build_model()
    
    def build_model(self):
        
        input_placeholder = Input(shape = self.input_shape)

        x = Conv2D(
            filters = 64,
            kernel_size = 9,
            strides = 1,
            padding = 'same'
        )(input_placeholder)
        x = PReLU(
            alpha_initializer = 'zeros',
            alpha_regularizer = None,
            alpha_constraint = None,
            shared_axes = [1, 2]
        )(x)

        skip = x

        for _ in range(16):
            x = Utils.residual_block(x, 64, 3, 1)
        
        x = Conv2D(
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = 'same'
        )(x)
        x = BatchNormalization(momentum = 0.5)(x)
        x = add([skip, x])

        for _ in range(2):
            x = Utils.upsampling_block(x, 256, 3, 1)
        
        x = Conv2D(
            filters = 3,
            kernel_size = 9,
            strides = 1,
            padding = 'same'
        )(x)
        output = Activation('tanh')(x)

        self.generator = Model(input_placeholder, output)
    
    def display(self):
        self.generator.summary()


class Discriminator(object):

    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.build_model()
    
    def build_model(self):

        input_placeholder = Input(shape = self.input_shape)

        x = Conv2D(
            filters = 64,
            kernel_size = 3,
            strides = 1,
            padding = 'same'
        )(input_placeholder)
        x = LeakyReLU(alpha = 0.2)(x)

        x = Utils.discriminator_block(x, 64, 3, 2)
        x = Utils.discriminator_block(x, 128, 3, 1)
        x = Utils.discriminator_block(x, 128, 3, 2)
        x = Utils.discriminator_block(x, 256, 3, 1)
        x = Utils.discriminator_block(x, 256, 3, 2)
        x = Utils.discriminator_block(x, 512, 3, 1)
        x = Utils.discriminator_block(x, 512, 3, 2)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha = 0.2)(x)

        output = Dense(1, activation = 'sigmoid')(x)

        self.discriminator = Model(input_placeholder, output)
    
    def display(self):
        self.discriminator.summary()
from tensorflow.keras.layers import (
    Conv2D, add,
    BatchNormalization,
    PReLU, LeakyReLU,
    Conv2DTranspose,
    UpSampling2D
)
from tqdm import tqdm
from glob import glob
import cv2


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

    @staticmethod
    def normalize(image):
        return (image.astype(np.float32) - 127.5) / 127.5

    @staticmethod
    def denormalize(image):
        image = (image + 1) * 127.5
        return image.astype(np.uint8)

    @staticmethod
    def load_data(data_directory, n_images):
        image_paths = glob(data_directory + '/*')
        x_train, y_train = [], []
        for i in tqdm(range(n_images)):
            image = cv2.imread(image_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Utils.normalize(cv2.resize(image, (384, 384)))
            x_train.append(image)
        x_train = np.array(x_train)
        for image in tqdm(x_train):
            y_train.append(cv2.resize(image, (96, 96)))
        y_train = np.array(y_train)
        return x_train, y_train
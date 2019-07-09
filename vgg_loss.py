from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19


class VGGLoss:


    def __init__(self, image_dimensions):
        self.image_dimensions = image_dimensions
    

    def loss(self, y_true, y_pred):
        """
        Computes content loss for Lower Resolution Images

        ...

        Parameters
        ----------
        y_true : tensor
            Actual Higher Resolution Image
        y_pred : tensor
            Predicted Higher Resolution Image
        
        Returns
        -------
        content loss using VGG19 model
        """
        vgg = VGG19(include_top = False, weights = 'imagenet', input_shape = self.image_dimensions)
        vgg.trainable = False
        for layer in vgg.layers:
            layer.trainable = False
        model = Model(vgg.input, vgg.get_layer('block5_conv4').output)
        model.trainable = False
        return K.mean(K.square(model(y_true) - model(y_pred)))
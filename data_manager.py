import numpy as np
from glob import  glob
from skimage.data import imread
from skimage.transform import resize


class DataManager:


    def __init__(self, high_res_dim, low_res_dim):
        self.high_resolution_dimension = high_res_dim
        self.low_resolution_dimension = low_res_dim
    

    def normalize(self, image):
        """
        Normalize an image

        Parameters
        ----------
        image : numpy tensor
            Image to be normalized
        
        Returns
        -------
        normalized image
        """
        return (image.astype(np.float32) - 127.5) / 127.5
    

    def denormalize(self, image):
        """
        De-normalize an image

        Parameters
        ----------
        image : numpy tensor
            Image to be de-normalized
        
        Returns
        -------
        De-normalized image
        """
        image = (image + 1) * 127.5
        return image.astype(np.uint8)
    

    def read_hr_image(self, image_path):
        """
        Read a Higher Resolution image

        Parameters
        ----------
        image : string
            Image Path
        
        Returns
        -------
        Normalized High Resolution image
        """
        hr_image = imread(image_path)
        hr_image = resize(hr_image, (self.high_resolution_dimension[0], self.high_resolution_dimension[1]))
        hr_image = self.normalize(hr_image)
        return hr_image
    

    def read_lr_image(self, image_path):
        """
        Read a Lower Resolution image

        Parameters
        ----------
        image : string
            Image Path
        
        Returns
        -------
        Normalized Low Resolution image
        """
        lr_image = imread(image_path)
        lr_image = resize(lr_image, (self.low_resolution_dimension[0], self.low_resolution_dimension[1]))
        lr_image = self.normalize(lr_image)
        return lr_image
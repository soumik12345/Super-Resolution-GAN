import numpy as np
from glob import  glob
from skimage.data import imread
from skimage.transform import resize
from tqdm import tqdm


class DataManager:


    def __init__(self, high_res_dim, low_res_dim, train_images_directory):
        self.high_resolution_dimension = high_res_dim
        self.low_resolution_dimension = low_res_dim
        self.train_images_directory = train_images_directory
    

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
    

    def load_training_images(self):
        """
        Load Training Images

        ...

        Returns
        ----------
        training_lr_images : numpy tensor
            numpy array of all low resolutuion training images (preprocessed)
        training_hr_images : numpy tensor
            numpy array of all high resolutuion training images (preprocessed)
        """
        images_list = glob(self.train_images_directory + '/*')
        training_hr_images, training_hr_images = [], []
        
        for image_path in tqdm(images_list):
            training_hr_images.append(self.read_hr_image(image_path))
            training_lr_images.append(self.read_lr_image(image_path))
        
        training_hr_images = np.array(training_hr_images)
        training_lr_images = np.array(training_lr_images)
        
        return training_lr_images, training_hr_images

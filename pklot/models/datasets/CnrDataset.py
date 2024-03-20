import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import PIL
from torchvision.io import read_image
from mmdet.registry import DATASETS


@DATASETS.register_module()
class CnrDataset(Dataset):
    """ CNR dataset

    Dataset file structure:
    - FULL_IMAGE_1000x750
        - OVERCAST
            - 2015-11-16
                - camera1
                - camera2
        - RAINY
            - 2016-01-08
                - camera1
                - camera2
        - SUNNY
            - 2015-11-12
                - camera1
                - camera2
    """

    def __init__(self, img_dir, transform=None, target_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
        """
        print("using CnrDataset")
        self.img_dir = img_dir
        self.parse_dir()
        self.transform = transform
        self.target_transform = target_transform

    def parse_dir(self):
        """ Enumerate all subdirectories and parse them.

        item = {
            "img_name": "2015-11-12_0709.jpg",
            "img_camera": "camera1",
            "img_path": "./SUNNY/2015-11-12/camera1/2015-11-12_0709.jpg",
            "img_weather": "SUNNY"
            }
        """
        self.images = []

        for weather in os.listdir(self.img_dir):
            weather_dir = os.path.join(self.img_dir, weather)
            if os.path.isdir(weather_dir):
                for date in os.listdir(weather_dir):
                    date_dir = os.path.join(weather_dir, date)
                    if os.path.isdir(date_dir):
                        for camera in os.listdir(date_dir):
                            camera_dir = os.path.join(date_dir, camera)
                            if os.path.isdir(camera_dir):
                                for img_name in os.listdir(camera_dir):
                                    img_path = os.path.join(camera_dir, img_name)
                                    item = {
                                        "img_name": img_name,
                                        "img_camera": camera,
                                        "img_path": img_path,
                                        "img_weather": weather,
                                    }
                                    self.images.append(item)

    def __len__(self):
        """ Return the length of the dataset. """
        print("len of images: ", len(self.images))
        return len(self.images)

    def __getitem__(self, idx):
        """ Return the item at index idx. """
        print("__getitem__: ", idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get the image path and label
        img_path = self.images[idx]["img_path"]
        img_camera = self.images[idx]["img_camera"]
        img_weather = self.images[idx]["img_weather"]

        # read the image
        #image = read_image(img_path)
        image = PIL.Image.open(img_path)

        # apply the image transform
        if self.transform:
            image = self.transform(image)

        # create the target
        target = {
            "camera": img_camera,
            "weather": img_weather,
            "path": img_path
        }

        # apply the target transform
        if self.target_transform:
            target = self.target_transform(target)

        return image, target
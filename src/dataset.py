import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2

# local files
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.config import config


class SETIDataset(Dataset):
    def __init__(self, df, mode="train", clip_rate=3.5):
        self.transform = self.__get_augmentation(mode)
        self.mode = mode
        self.df = df
        self.clip_rate = clip_rate

    def __getitem__(self, idx):
        image = np.load(self.df.loc[idx, "file_path"])
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0))
        # image = np.vstack([image[0], image[2], image[4]]).transpose((1, 0)) # ON Target
        image = cv2.resize(image, (config.image_size[1], config.image_size[0])) # cv2.resize(image, (width, height))
        mean = np.mean(image)
        std = np.std(image)
        image = np.clip(image, mean-self.clip_rate*std, mean+self.clip_rate*std)
        image = image[:, :, np.newaxis]
        image = self.__data_augmentation(self.transform, image)

        if self.mode == "test":
            return image
        else:
            label = torch.tensor(self.df.loc[idx, "target"]).float()
            return image, label

    def __len__(self):
        return self.df.shape[0]

    def __get_augmentation(self, mode="train"):
        if mode == "train":
            transform = [
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                ToTensorV2(),
            ]
        else:
            transform = [
                ToTensorV2(),
            ]
        return albu.Compose(transform)

    def __data_augmentation(self, transform, image):
        augmented = transform(image=image)
        image = augmented['image']
        return image


# def process_image(image, threshold=3.5):
#     # clip intensities
#     mean = np.mean(image)
#     std = np.std(image)
#     image = np.clip(image, mean-threshold*std, mean+threshold*std)
#     # morph close
#     morphed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=np.ones((3, 3), dtype=np.float32))
#     # gradient in both directions
#     sobelx = cv2.Sobel(morphed, cv2.CV_64F, 1, 0, 2)
#     sobely = cv2.Sobel(morphed, cv2.CV_64F, 0, 1, 2)
#     # blend
#     blended = cv2.addWeighted(src1=sobelx, alpha=0.7, src2=sobely, beta=0.3, gamma=0)
#     return blended

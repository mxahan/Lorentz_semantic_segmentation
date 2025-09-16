import os
import random
import time
import numpy as np
import torch
from PIL import Image
import yaml

from torchvision import transforms, datasets
from torchvision.transforms import functional as Ft
from torchvision.transforms import InterpolationMode

import glob


import torch.nn.functional as F

from data.ade20k import Rotate, Shear, Skew, Crop, GaussianNoise, SaltAndPepperNoise, MotionBlur, HideAndSeekNoise

class ADE20k_challange(torch.utils.data.Dataset):
    def __init__(self, path_list = 'dataset/ADEChallengeData2016/' , transform = None, data_set = 'training', seed = None, aug = True,
                label_convert = False, img_size=768, return_label = True,  
                interpolation=Image.NEAREST, color_pallete = 'city', norm_ = False):
        """
        :param path_list: Path to file listing image paths.
        :param transform: Additional torchvision transforms.
        :param data_set: 'train' or other mode.
        :param seed: Seed for shuffling.
        :param img_size: Resize dimensions.
        :param interpolation: Interpolation method for resizing.
        """
        self.transform = transform
        if data_set in ['training', 'validation']:
            self.tr_val = data_set # 'training' or 'validation'
        else: raise Exception("data_set should be 'training' or 'validation'")
        
        image_dir = os.path.join(path_list, 'images/',self.tr_val)
        # label_dir = os.path.join('dataset/ADEChallengeData2016/', 'annotations/',tr_val)
        self.imgs = glob.glob(os.path.join(image_dir,"*.jpg"))
        
        self.color_pallete = color_pallete
        self.norm_ = norm_
        if seed:
            random.seed(seed)
            random.shuffle(self.imgs)
            
        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]
        self.normalize = transforms.Normalize(self.MEAN, self.STD)

        self.masks = [
            path.replace(".jpg", ".png").replace('images', 'annotations')
            for path in self.imgs
        ]
        self.learning_map = None
        
        self.aug_weights = [4, 0.3, 0.3, 0.2, 0.2, 0.05, 0.05, 0.02, 0.02]
        if img_size:
            self.transform_resize = transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)

        self.aug = aug
        self.label_convert = label_convert

        
        self.to_tensor = transforms.ToTensor()

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        converted_label = np.zeros_like(label)
        for k, v in self.learning_map.items():
            converted_label[temp == k] = v
        return converted_label

    def get_color_pallete(self, npimg, dataset='city'):
        out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
        if dataset == 'city':
            cityspallete = [
                0, 0, 0,
                128, 64, 128,
                244, 35, 232,
                70, 70, 70,
                102, 102, 156,
                190, 153, 153,
                153, 153, 153,
                250, 170, 30,
                220, 220, 0,
                107, 142, 35,
                152, 251, 152,
                0, 130, 180,
                220, 20, 60,
                255, 0, 0,
                0, 0, 142,
                0, 0, 70,
                0, 60, 100,
                0, 80, 100,
                0, 0, 230,
                119, 11, 32,
            ]
            out_img.putpalette(cityspallete)
        else:
            vocpallete = _getvocpallete(256)
            out_img.putpalette(vocpallete)
        return out_img.convert("RGB")

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

        # Load and resize the image

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        
        if self.norm_: img = self.normalize(self.to_tensor(img))
        # normalization returns 3,h, w image and different range than the 0-1 or unit
        
        return img, mask, img_path

    def __len__(self):
        return len(self.imgs)
    
'''

  def __getitem__(self, index):
        img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()

        # Load and resize the image

        img = Image.open(img_path).convert('RGB')
        # img = self.transform_resize(img)
        # Load, convert, and resize the mask
        mask = Image.open(mask_path)

        # mask = self.transform_resize(mask)
#         if self.label_convert:
#             mask = self.convert_label(mask)
#             mask = mask.astype(np.uint8)
#             mask = self.get_color_pallete(mask, self.color_pallete)
       
        # mask = Image.fromarray(mask)
        # Augmentation stage
        
        
        # augmentation_num = random.choices(range(9), weights=self.aug_weights, k=1)[0] if self.aug else 0
        # if augmentation_num > 0:
        #     augmentation_set = [
        #         transforms.RandomHorizontalFlip(p=1),  # Flip horizontally
        #         transforms.RandomVerticalFlip(p=1),  # Flip vertically
        #         Crop(min_crop=0.6, max_crop=0.9),  # Random crop
        #         Rotate(angle=90),  # Rotate
        #         Shear(shear=10, scale=(0.8, 1.2)),  # Shear
        #         Skew(magnitude=0.2),  # Skew
        #         HideAndSeekNoise(min_size=20, max_size=100), #Hide and seek noise
        #         GaussianNoise(mean=0, std=(5,20)), # Gaussian noise (only for image) / std 10-20
        #         SaltAndPepperNoise(min_prob=0.01, max_prob=0.03), #Salt and pepper noise (only for image) 
        #         transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
        #         MotionBlur(min_size=3, max_size=15), # Motion blur (only for image)
        #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter (only for image)
        #     ]
        #     random.shuffle(augmentation_set)
        #     augmentation_set = augmentation_set[:augmentation_num]
        #     for aug in augmentation_set:
        #         if isinstance(aug, (transforms.GaussianBlur, transforms.ColorJitter, GaussianNoise, SaltAndPepperNoise, MotionBlur)):
        #             img = aug(img)
        #         else:
        #             img = aug(img)
        #             mask = aug(mask)


        

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        
        if self.norm_: img = self.normalize(self.to_tensor(img))
        # normalization returns 3,h, w image and different range than the 0-1 or unit
        
        return img, mask, img_path

    def __len__(self):
        return len(self.imgs)
'''
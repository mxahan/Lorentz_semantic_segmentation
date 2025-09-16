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


import torch.nn.functional as F

from data.ade20k import Rotate, Shear, Skew, Crop, GaussianNoise_torch, HideAndSeekNoise_torch, SaltAndPepperNoise_torch, MotionBlur_torch
from data.ade20k import HideAndSeekNoise_torch_gpu, MotionBlur_torch_gpu, SaltAndPepperNoise_torch_gpu, GaussianNoise_torch_gpu, Rotate_sync
from data.ade20k import FlipHorizontal_sync, FlipVertical_sync, Crop_sync, Shear_sync, Skew_sync

def _augmentation_(batch_tensor, batch_target, augmentation_num = 1, im_size = 1024):  
    """
    Applies a sequence of randomized augmentations to a batch of images and their corresponding semantic segmentation masks.

    Args:
        batch_tensor (Tensor): A tensor of shape [B, C, H, W] representing a batch of input images.
        batch_target (Tensor): A tensor of shape [B, H, W] representing the corresponding target labels (segmentation masks).
        augmentation_num (int): The number of augmentations to randomly select and apply from the predefined set.
        im_size (int): The target size to which the augmented images and masks will be resized.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Augmented images of shape [B, C, im_size, im_size], dtype uint8.
            - Augmented targets of shape [B, im_size, im_size], dtype long.
    """
    
    augmentation_set = [
                transforms.RandomHorizontalFlip(p=1),  # Flip horizontally
                transforms.RandomVerticalFlip(p=1),  # Flip vertically
                Crop(min_crop=0.6, max_crop=0.9),   # Random crop
                Rotate(angle=90),  # Rotate
                Shear(shear=10, scale=(0.8, 1.2)),  # Shear
                Skew(magnitude=0.2),  # Skew
                HideAndSeekNoise_torch(min_size=90, max_size=150), #Hide and seek noise
                GaussianNoise_torch(mean=0, std=(5,20)), # Gaussian noise (only for image) / std 10-20
                SaltAndPepperNoise_torch(min_prob=0.01, max_prob=0.03), #Salt and pepper noise (only for image) 
                transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
                MotionBlur_torch(min_size=3, max_size=15), # Motion blur (only for image)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter (only for image)
            ]
    transform_train = transforms.Compose([
        transforms.Resize((im_size, im_size), interpolation=InterpolationMode.NEAREST),  # Resize the image
        transforms.PILToTensor()  # Convert to uint8 tensor
        ])
    
    
    
    random.shuffle(augmentation_set)
    augmentation_set = augmentation_set[:augmentation_num]
    
    bs = batch_tensor.size(0)
    augmented_batch, augmented_targets = [], []

    for i in range(bs):
        img = Ft.to_pil_image(batch_tensor[i])  # Convert to PIL
        target = Ft.to_pil_image(batch_target[i].byte()) # [H, W] → PIL (must be mode="L")
        
        if augmentation_num>0:
                random.shuffle(augmentation_set)
                augmentation_set = augmentation_set[:augmentation_num]
                for aug in augmentation_set:
                    if isinstance(aug, (transforms.GaussianBlur, transforms.ColorJitter, GaussianNoise_torch, SaltAndPepperNoise_torch, MotionBlur_torch)):
                        img = aug(img)
                    else:
                        img = aug(img)
                        target = aug(target)
                        

        img_tensor = transform_train(img)  # Convert back to tensor
        taget_tensor  = transform_train(target).squeeze()
        
        augmented_batch.append(img_tensor)
        augmented_targets.append(taget_tensor) 

    return torch.stack(augmented_batch), torch.stack(augmented_targets)


class ImageOnlyTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, batch_img, batch_target=None):
        """
        Args:
            batch_img: Tensor of shape (B, C, H, W)
            batch_target: Tensor of shape (B, H, W) or (B, 1, H, W), optional

        Returns:
            (transformed_img, unchanged_target)
        """
        B = batch_img.size(0)
        transformed_imgs = []

        for i in range(B):
            img = batch_img[i]  # (C, H, W)
            transformed_img = self.transform(img)  # Assume transform works on (C, H, W) tensor
            transformed_imgs.append(transformed_img)

        transformed_batch = torch.stack(transformed_imgs)

        return transformed_batch, batch_target  # target unchanged
    
class ImageMaskTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, mask):
        return self.transform(img), self.transform(mask)
    
    
def _augmentation_gpu(batch_tensor, batch_target, augmentation_num = 1, im_size = 1024):  
    """
    Applies a sequence of randomized augmentations to a batch of images and their corresponding semantic segmentation masks.

    Args:
        batch_tensor (Tensor): A tensor of shape [B, C, H, W] representing a batch of input images.
        batch_target (Tensor): A tensor of shape [B, H, W] representing the corresponding target labels (segmentation masks).
        augmentation_num (int): The number of augmentations to randomly select and apply from the predefined set.
        im_size (int): The target size to which the augmented images and masks will be resized.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing:
            - Augmented images of shape [B, C, im_size, im_size], dtype uint8.
            - Augmented targets of shape [B, im_size, im_size], dtype long.
    """
    device = batch_tensor.device 
    
    augmentation_set = [
                FlipHorizontal_sync(p=1.0),
                FlipVertical_sync(p=1.0),
                # Crop_sync(min_crop=0.6, max_crop=0.9),   # Random crop
                Rotate_sync(max_angle= 15),  # Rotate
                # Shear_sync(shear=10, scale=(0.8, 1.2)),  # Shear
                # Skew_sync(magnitude=0.2),  # Skew
                HideAndSeekNoise_torch_gpu(patch_size_range=(20, 30)), #Hide and seek noise
        
                GaussianNoise_torch_gpu(mean=0, std_range=(5,20)), # Gaussian noise (only for image) / std 10-20
                SaltAndPepperNoise_torch_gpu(prob_range=(0.01, 0.03)), #Salt and pepper noise (only for image) 
                ImageOnlyTransform(transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1))),  # Gaussian blur (only for image)
                MotionBlur_torch_gpu(kernel_size_range=(3, 15)), # Motion blur (only for image)
                # ImageOnlyTransform(transforms.ColorJitter(brightness=0.2, hue = 0.2)),
            ]
    
    
    
    random.shuffle(augmentation_set)
    selected_augs = augmentation_set[:augmentation_num]
    
    # print(selected_augs)
    
    
    if len(selected_augs)>0:
        for aug in selected_augs:
            batch_tensor, batch_target = aug(batch_tensor, batch_target)
        
    
    if batch_target.ndim == 4: 
        batch_target = batch_target.squeeze(1)
        
    if batch_tensor.shape[-2:] != (im_size, im_size):
        batch_tensor = F.interpolate(batch_tensor, size=(im_size, im_size), mode='bilinear', align_corners=False)
        batch_target = F.interpolate(batch_target.unsqueeze(1).float(), size=(im_size, im_size), mode='nearest').squeeze(1).long()
        
    return batch_tensor.int().to(device), batch_target.int().to(device)
    


    
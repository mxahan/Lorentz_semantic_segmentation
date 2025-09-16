import os
import random
import time
import numpy as np
import torch
from PIL import Image
import cv2
import random
import math

from torchvision import transforms, datasets
from torchvision.transforms import functional as Ft
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F

def _getvocpallete(num_colors):
    return [0, 0, 0] * num_colors


# ----------------------
# Augmentation Classes
# ----------------------

class FlipHorizontal_sync:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-1])  # flip width
            mask = torch.flip(mask, dims=[-1])
        return img, mask

class FlipVertical_sync:
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[-2])  # flip height
            mask = torch.flip(mask, dims=[-2])
        return img, mask
    
    
class Rotate:
    def __init__(self, angle):
        self.angle = random.randint(-angle, angle)

    def __call__(self, img):
        return Ft.rotate(img, angle=self.angle)
    
class Rotate_sync:
    def __init__(self, max_angle=30.0):
        self.max_angle = max_angle  

    def __call__(self, img, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # ensure mask shape [B, 1, H, W]

        B, C, H, W = img.shape
        theta_list = []

        for _ in range(B):
            angle = random.uniform(-self.max_angle, self.max_angle)
            angle_rad = math.radians(angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            theta = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0]
            ], dtype=torch.float32)
            theta_list.append(theta)

        theta_batch = torch.stack(theta_list).to(img.device)  # [B, 2, 3]

        grid = F.affine_grid(theta_batch, size=img.size(), align_corners=False)
        img_rot = F.grid_sample(img.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)
        mask_rot = F.grid_sample(mask.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)

        return img_rot, mask_rot
    
class Shear:
    def __init__(self, shear=10, scale=(1.0, 1.0)):
        self.shear = random.uniform(-shear, shear)
        self.scale = random.uniform(scale[0], scale[1])

    def __call__(self, img):
        return Ft.affine(img, angle=0, translate=(0, 0), scale=self.scale, shear=[self.shear, self.shear])

class Shear_sync:
    def __init__(self, shear=10.0, scale=(1.0, 1.0)):
        self.shear = shear
        self.scale = scale

    def __call__(self, img, mask):
        B, C, H, W = img.shape
        theta_list = []

        for _ in range(B):
            shear_value = random.uniform(-self.shear, self.shear) / 100.0  # convert to proportion
            scale_value = random.uniform(self.scale[0], self.scale[1])

            theta = torch.tensor([
                [1.0, shear_value, 0.0],
                [0.0, scale_value, 0.0]
            ], dtype=torch.float32)
            theta_list.append(theta)

        theta_batch = torch.stack(theta_list).to(img.device)  # Shape: [B, 2, 3]

        # Affine grid should match input shape
        grid = F.affine_grid(theta_batch, size=img.size(), align_corners=False)

        img_out = F.grid_sample(img.float(), grid, mode='nearest', padding_mode='border', align_corners=False)
        mask_out = F.grid_sample(mask.float(), grid, mode='nearest', padding_mode='border', align_corners=False)

        return img_out, mask_out
    
    
class Skew:
    def __init__(self, magnitude=0.2):
        self.xshift = random.uniform(-magnitude, magnitude)
        self.yshift = random.uniform(-magnitude, magnitude)

    def __call__(self, img):
        width, height = img.size
        x_shift = int(self.xshift * width)
        y_shift = int(self.yshift * height)
        return img.transform(img.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))


class Skew_sync:
    def __init__(self, magnitude=0.2):
        self.magnitude = magnitude
        
    def __call__(self, img, mask):
        B, C, H, W = img.shape
        theta_list = []

        for _ in range(B):
            x_skew = random.uniform(-self.magnitude, self.magnitude)
            y_skew = random.uniform(-self.magnitude, self.magnitude)

            # Skew matrix: affects the affine transform
            theta = torch.tensor([
                [1.0, x_skew, 0.0],
                [y_skew, 1.0, 0.0]
            ], dtype=torch.float32)
            theta_list.append(theta)

        theta_batch = torch.stack(theta_list).to(img.device)  # Shape: [B, 2, 3]

        grid = F.affine_grid(theta_batch, size=img.size(), align_corners=False)

        img_out = F.grid_sample(img.float(), grid, mode='nearest', padding_mode='border', align_corners=False)
        mask_out = F.grid_sample(mask.float(), grid, mode='nearest', padding_mode='border', align_corners=False)

        return img_out, mask_out

    
    
class Crop:
    def __init__(self, min_crop=0.8, max_crop=0.9):
        self.crop_scale = random.uniform(min_crop, max_crop)
        self.seed = time.time()

    def __call__(self, img):
        width, height = img.size        
        crop_width = int(self.crop_scale * width)
        crop_height = int(self.crop_scale * height)
        
        random.seed(self.seed)
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        return Ft.crop(img, top, left, crop_height, crop_width).resize((width, height), resample=Image.NEAREST)

class Crop_sync:
    def __init__(self, min_crop=0.8, max_crop=0.9):
        self.min_crop = min_crop
        self.max_crop = max_crop

    def __call__(self, img, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # Ensure mask shape [B, 1, H, W]

        B, C, H, W = img.shape
        crop_h, crop_w = int(H * self.crop_size), int(W * self.crop_size)

        img_out = []
        mask_out = []

        for i in range(B):
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)

            img_crop = img[i, :, top:top + crop_h, left:left + crop_w]
            mask_crop = mask[i, :, top:top + crop_h, left:left + crop_w]

            img_resized = F.interpolate(img_crop.unsqueeze(0).float(), size=(H, W), mode='nearest', align_corners=False)
            mask_resized = F.interpolate(mask_crop.unsqueeze(0).float(), size=(H, W), mode='nearest')

            img_out.append(img_resized)
            mask_out.append(mask_resized)

        img_out = torch.cat(img_out, dim=0)
        mask_out = torch.cat(mask_out, dim=0)

        return img_out, mask_out


class GaussianNoise:
    def __init__(self, mean=0, std=(10,20)):
        self.mean = mean
        self.std = random.uniform(std[0], std[1])

    def __call__(self, img):
        img = np.array(img)

        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)
    
class GaussianNoise_torch:
    def __init__(self, mean=0, std=(10,20)):
        self.mean = mean
        self.std = random.uniform(std[0], std[1])

    def __call__(self, img):
        # img: PIL Image → convert to float tensor [0, 255]
        img_tensor = Ft.to_tensor(img) * 255.0  # [C, H, W], float32
        # Add Gaussian noise
        noise = torch.randn_like(img_tensor) * self.std + self.mean
        noisy_img = img_tensor + noise
        # Clamp and convert back to uint8
        noisy_img = torch.clamp(noisy_img, 0, 255).to(torch.uint8)
        # Convert to PIL
        return Ft.to_pil_image(noisy_img)

class GaussianNoise_torch_gpu:
    def __init__(self, mean=0, std_range=(5, 20)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, img_tensor, mask_tensor):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(img_tensor) * std + self.mean
        img_tensor = torch.clamp(img_tensor + noise, 0, 255)
        return img_tensor, mask_tensor
    

class SaltAndPepperNoise:
    def __init__(self, min_prob=0.01, max_prob=0.05):
        self.salt_prob = random.uniform(min_prob, max_prob)
        self.pepper_prob = random.uniform(min_prob, max_prob)

    def __call__(self, img):
        img_array = np.array(img)

        salt_mask = np.random.rand(*img_array.shape[:2]) < self.salt_prob
        pepper_mask = np.random.rand(*img_array.shape[:2]) < self.pepper_prob
        img_array[salt_mask] = 255
        img_array[pepper_mask] = 0
        return Image.fromarray(img_array.astype(np.uint8))

class SaltAndPepperNoise_torch:
    def __init__(self, min_prob=0.01, max_prob=0.05):
        self.salt_prob = random.uniform(min_prob, max_prob)
        self.pepper_prob = random.uniform(min_prob, max_prob)

    def __call__(self, img):
        # Convert PIL to uint8 tensor in [0, 255]
        img_tensor = Ft.pil_to_tensor(img)  # [C, H, W]

        _, H, W = img_tensor.shape
        salt_mask = (torch.rand(H, W) < self.salt_prob)
        pepper_mask = (torch.rand(H, W) < self.pepper_prob)

        # Expand masks to shape [C, H, W]
        salt_mask = salt_mask.expand_as(img_tensor)
        pepper_mask = pepper_mask.expand_as(img_tensor)

        # Apply salt (255) and pepper (0)
        img_tensor[salt_mask] = 255
        img_tensor[pepper_mask] = 0
        return Ft.to_pil_image(img_tensor)
    
    
class SaltAndPepperNoise_torch_gpu:
    def __init__(self, prob_range=(0.01, 0.03)):
        self.prob_range = prob_range

    def __call__(self, img_tensor, mask_tensor):
        B, C, H, W = img_tensor.shape
        salt_prob = random.uniform(*self.prob_range)
        pepper_prob = random.uniform(*self.prob_range)

        salt_mask = torch.rand(B, 1, H, W, device=img_tensor.device) < salt_prob
        pepper_mask = torch.rand(B, 1, H, W, device=img_tensor.device) < pepper_prob

        salt_mask = salt_mask.expand(-1, C, -1, -1)
        pepper_mask = pepper_mask.expand(-1, C, -1, -1)

        img_tensor = img_tensor.clone()
        img_tensor[salt_mask] = 255
        img_tensor[pepper_mask] = 0
        return img_tensor, mask_tensor
    
class MotionBlur:
    def __init__(self, min_size=3, max_size=21):
        self.kernel_size = random.randint(min_size, max_size)

    def __call__(self, img):
        img_array = np.array(img)

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size
        blurred = cv2.filter2D(img_array, -1, kernel)
        return Image.fromarray(blurred.astype(np.uint8))
    
class MotionBlur_torch:
    def __init__(self, min_size=3, max_size=21):
        self.kernel_size = random.randint(min_size, max_size)

    def __call__(self, img):
        # Convert PIL to uint8 tensor [C, H, W]
        img_tensor = Ft.pil_to_tensor(img).float().unsqueeze(0)  # [1, C, H, W]

        # Create horizontal blur kernel [1, 1, 1, k]
        kernel = torch.ones(1, 1, 1, self.kernel_size) / self.kernel_size
        kernel = kernel.to(img_tensor.device)

        C = img_tensor.shape[1]
        kernel = kernel.expand(C, 1, 1, self.kernel_size)  # [C,1,1,k]

        # Apply convolution with padding to maintain size
        padding = (self.kernel_size // 2, 0)
        blurred = F.conv2d(img_tensor, kernel, padding=padding, groups=C)

        # Convert back to uint8 PIL
        blurred = blurred.squeeze(0).clamp(0, 255).to(torch.uint8)
        return Ft.to_pil_image(blurred)
    
class MotionBlur_torch_gpu:
    def __init__(self, kernel_size_range=(3, 15)):
        self.kernel_size = random.choice(range(*kernel_size_range))

    def __call__(self, img_tensor, mask_tensor):
        B, C, H, W = img_tensor.shape
        k = self.kernel_size
        padding = (k // 2, 0)

        kernel = torch.ones(1, 1, 1, k, device=img_tensor.device) / k
        kernel = kernel.expand(C, 1, 1, k)

        blurred = F.conv2d(img_tensor.float(), kernel, padding=padding, groups=C)
        return blurred.clamp(0, 255), mask_tensor
    
class HideAndSeekNoise:
    def __init__(self, min_size=90, max_size=190):
        self.patch_size = random.randint(min_size, max_size)
        self.seed = time.time()

    def __call__(self, img):
        img_array = np.array(img)
        len_var =  len(img_array.shape)
        if  len_var == 2: img_array = img_array[..., np.newaxis];
            
        height, width, _ = img_array.shape
        
        random.seed(self.seed)
        
        if self.patch_size>(height-20):
            self.patch_size = height-20
        if self.patch_size>(width-20):
            self.patch_size = width-20
        
        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)

        if len_var == 3:
            img_array[top:top + self.patch_size, left:left + self.patch_size] = [0, 0, 0]
        else: 
            img_array[top:top + self.patch_size, left:left + self.patch_size] = [0]
            img_array = img_array.squeeze(-1)
            
        return Image.fromarray(img_array)
    
class HideAndSeekNoise_torch_gpu:
    def __init__(self, patch_size_range=(90, 150)):
        self.patch_size = random.randint(*patch_size_range)

    def __call__(self, img_tensor, mask_tensor):
        B, C, H, W = img_tensor.shape
        patch_size = min(self.patch_size, H - 20, W - 20)

        for i in range(B):
            top = random.randint(0, H - patch_size)
            left = random.randint(0, W - patch_size)
            img_tensor[i, :, top:top + patch_size, left:left + patch_size] = 0

        return img_tensor, mask_tensor
    
    
class HideAndSeekNoise_torch:
    def __init__(self, min_size=90, max_size=190):
        self.patch_size = random.randint(min_size, max_size)
        self.seed = time.time()

    def __call__(self, img):
        is_gray = img.mode == 'L'

        # Convert to tensor (uint8): [C, H, W]
        img_tensor = Ft.pil_to_tensor(img)  # dtype: uint8
        C, H, W = img_tensor.shape

        random.seed(self.seed)

        # Clamp patch size if too big
        patch_size = min(self.patch_size, H - 20, W - 20)

        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)

        # Zero-out patch
        img_tensor[:, top:top + patch_size, left:left + patch_size] = 0

        return Ft.to_pil_image(img_tensor)
    
class ADE20k(torch.utils.data.Dataset):
    def __init__(self, path_list, transform = None, data_set = 'val', seed = None, aug = True,
                label_convert = False, img_size=768, return_label = True,  
                interpolation=Image.NEAREST, color_pallete = 'city'):
        """
        :param path_list: Path to file listing image paths.
        :param transform: Additional torchvision transforms.
        :param data_set: 'train' or other mode.
        :param seed: Seed for shuffling.
        :param img_size: Resize dimensions.
        :param interpolation: Interpolation method for resizing.
        """
        self.transform = transform
        self.data_set = data_set
        self.color_pallete = color_pallete

        with open(path_list, "r") as file:
            self.imgs = file.readlines()

        if seed:
            random.seed(seed)
            random.shuffle(self.imgs)

        self.masks = [
            path.replace(".jpg", "_seg.png")
            for path in self.imgs
        ]
        self.learning_map = None
        
        self.aug_weights = [5, 0.3, 0.3, 0.2, 0.2, 0.05, 0.05, 0.02, 0.02]
        if img_size:
            self.transform_resize = transforms.Resize((img_size, img_size), interpolation=Image.NEAREST)

        self.aug = aug
        self.label_convert = label_convert
        self.return_label = return_label

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
        # img = self.transform_resize(img)
        # Load, convert, and resize the mask
        mask = Image.open(mask_path)

        # mask = self.transform_resize(mask)
        if self.label_convert:
            mask = self.convert_label(mask)
            mask = mask.astype(np.uint8)
            mask = self.get_color_pallete(mask, self.color_pallete)
       
        # mask = Image.fromarray(mask)
        # Augmentation stage
        augmentation_num = random.choices(range(9), weights=self.aug_weights, k=1)[0] if self.aug else 0
        if augmentation_num > 0:
            augmentation_set = [
                transforms.RandomHorizontalFlip(p=1),  # Flip horizontally
                transforms.RandomVerticalFlip(p=1),  # Flip vertically
                Crop(min_crop=0.6, max_crop=0.9),  # Random crop
                Rotate(angle=90),  # Rotate
                Shear(shear=10, scale=(0.8, 1.2)),  # Shear
                Skew(magnitude=0.2),  # Skew
                HideAndSeekNoise(min_size=90, max_size=150), #Hide and seek noise
                GaussianNoise(mean=0, std=(5,20)), # Gaussian noise (only for image) / std 10-20
                SaltAndPepperNoise(min_prob=0.01, max_prob=0.03), #Salt and pepper noise (only for image) 
                transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
                MotionBlur(min_size=3, max_size=15), # Motion blur (only for image)
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Color jitter (only for image)
            ]
            random.shuffle(augmentation_set)
            augmentation_set = augmentation_set[:augmentation_num]
            for aug in augmentation_set:
                if isinstance(aug, (transforms.GaussianBlur, transforms.ColorJitter, GaussianNoise, SaltAndPepperNoise, MotionBlur)):
                    img = aug(img)
                else:
                    img = aug(img)
                    mask = aug(mask)


        
        
        if self.return_label:
            H, W = mask.size
            mask = np.asarray(mask, dtype=np.int32).reshape(-1, 3)
            mask = np.linalg.norm(mask[:, None, :] - self.color_pallete[None, :, :], axis=2)
            mask = np.argmin(mask, axis=1)
            mask = mask.reshape(H, W)
        
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        
        
        return img, mask, img_path

    def __len__(self):
        return len(self.imgs)
    
'''
Extra codes: using recent ADE20K dataset
this is great for instance segmentation and objectclass segmentation. 

file = 'dataset/ADE20K_dataset/ADE20K_2021_17_01/images/ADE/training/nature_landscape/beach/ADE_train_00003089.jpg'
label = Image.open(file)
plt.imshow(label)



from PIL import Image
import matplotlib._color_data as mcd
import cv2
import json
import numpy as np
import os

fileseg = file.replace('.jpg', '_seg.png');
with Image.open(fileseg) as io:
    seg = np.array(io);
R = seg[:,:,0];
G = seg[:,:,1];
B = seg[:,:,2];
ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32));
Minstances_hat = np.unique(B, return_inverse=True)[1]
Minstances_hat = np.reshape(Minstances_hat, B.shape)
ObjectInstanceMasks = Minstances_hat


level = 0
PartsClassMasks = [];
PartsInstanceMasks = [];
while True:
    level = level+1;
    file_parts = file.replace('.jpg', '_parts_{}.png'.format(level));
    if os.path.isfile(file_parts):
        with Image.open(file_parts) as io:
            partsseg = np.array(io);
        R = partsseg[:,:,0];
        G = partsseg[:,:,1];
        B = partsseg[:,:,2];
        PartsClassMasks.append((np.int32(R)/10)*256+np.int32(G));
        PartsInstanceMasks = PartsClassMasks
        # TODO:  correct partinstancemasks
        print(level)

    else:
        break

objects = {}
parts = {}

attr_file_name = file.replace('.jpg', '.json')
if os.path.isfile(attr_file_name):
    with open(attr_file_name, 'r') as f:
        input_info = json.load(f)

    contents = input_info['annotation']['object']
    instance = np.array([int(x['id']) for x in contents])
    names = [x['raw_name'] for x in contents]
    corrected_raw_name =  [x['name'] for x in contents]
    partlevel = np.array([int(x['parts']['part_level']) for x in contents])
    ispart = np.array([p>0 for p in partlevel])
    iscrop = np.array([int(x['crop']) for x in contents])
    listattributes = [x['attributes'] for x in contents]
    polygon = [x['polygon'] for x in contents]
    for p in polygon:
        p['x'] = np.array(p['x'])
        p['y'] = np.array(p['y'])

    objects['instancendx'] = instance[ispart == 0]
    objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
    objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
    objects['iscrop'] = iscrop[ispart == 0]
    objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
    objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]


    parts['instancendx'] = instance[ispart == 1]
    parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
    parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
    parts['iscrop'] = iscrop[ispart == 1]
    parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
    parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

plt.imshow(ObjectClassMasks==2420)

plt.imshow(ObjectInstanceMasks==id) # the id is in the len(contents)


# the contend is a list of index that contain class information

contents[4]['raw_name'] # explore more of this variable


'''
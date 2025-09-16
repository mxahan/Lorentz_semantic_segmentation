import yaml
from PIL import Image
from data.base_dataset import BaseDataset

class CityScapes(BaseDataset):
    def __init__(self, path_list, config_file = "./dataset/CityScapes/cityscape.yaml", transform = None,
                data_set = 'val', seed=None, img_size=768, interpolation=Image.BILINEAR, color_pallete = 'city'):
        super().__init__(path_list, transform, data_set, seed, img_size, interpolation, color_pallete)
        with open(config_file, 'r') as stream:
            cityyaml = yaml.safe_load(stream)
        self.learning_map = cityyaml['learning_map']

        self.masks = [
            path.replace("/leftImg8bit/", "/gtFine/")
            .replace("_leftImg8bit.", "_gtFine_labelIds.")
            for path in self.imgs
        ]

'''
Extra Code: Citiscape dataset
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, path_list, transform = None, data_set = 'train'):
#         self.transform = transform
#         self.data_set = data_set

#         with open(path_list, "r") as file:
#             self.imgs = file.readlines()

#         self.masks = [
#             path.replace("/leftImg8bit/", "/gtFine/")
#             .replace("_leftImg8bit.", "_gtFine_labelIds.")
#             for path in self.imgs
#         ]


#         with open("./dataset/CityScapes/cityscape_copy.yaml", 'r') as stream:
#             cityyaml = yaml.safe_load(stream)
#         self.learning_map = cityyaml['learning_map']
        
#         self.aug_weights = [0.4, 0.3, 0.3, 0.2, 0.2, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01]
# #         print(self.imgs)

#     def convert_label(self, label, inverse=False):
#         temp = label.copy()
#         converted_label = np.zeros_like(label)
#         for k, v in self.learning_map.items():
#             converted_label[temp == k] = v
#         return converted_label

#     def get_color_pallete(self, npimg, dataset='voc'):
#         out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
#         if dataset == 'city':
#             cityspallete = cp_list_cp
#             out_img.putpalette(cityspallete)
#         else:
#             vocpallete = _getvocpallete(256)
#             out_img.putpalette(vocpallete)
#         return out_img.convert("RGB")

#     def __getitem__(self, index):
#         img_path, mask_path = self.imgs[index].rstrip(), self.masks[index].rstrip()
# #         img_path = self.imgs[index].rstrip()

#         img = Image.open(img_path).convert('RGB')
        
#         mask = Image.open(mask_path)
#         mask = np.array(mask)
#         mask = self.convert_label(mask)
#         mask = mask.astype(np.uint8)
#         mask = self.get_color_pallete(mask, "city")
        
        
#         augmentation_num = random.choices(range(11), weights=self.aug_weights, k=1)[0] if self.data_set == 'train' else 0
        
        
#         if augmentation_num > 0:
#             augmentation_set = [
#                 transforms.RandomHorizontalFlip(p=1),  # Flip horizontally
#                 transforms.RandomVerticalFlip(p=1),  # Flip vertically
#                 Crop(min_crop=0.7, max_crop=0.9),  # Random crop
#                 Rotate(angle=180),  # Rotate
#                 GaussianNoise(mean=0, std = 2), # Gaussian noise (only for image)
#                 SaltAndPepperNoise(salt_prob=0.1, pepper_prob=0.1), # Salt and pepper noise
#                 transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1)),  # Gaussian blur (only for image)
#                 Shear(shear=10, scale=(0.8, 1.2)),  # Shear
#                 Skew(magnitude=0.1),  # Skew
#                 transforms.ColorJitter(brightness=0.8, contrast=0.2, saturation=0.2),  # Color jitter (only for image)
#             ]
#             random.shuffle(augmentation_set)
#             augmentation_set = augmentation_set[:augmentation_num]
# #             augmentation_set = augmentation_set[4:5]
# #             Idx = [2]
# #             augmentation_set= list(map(augmentation_set.__getitem__, Idx))
        
#             for aug in augmentation_set:
#                 if isinstance(aug, (transforms.GaussianBlur, transforms.ColorJitter, GaussianNoise, SaltAndPepperNoise)):
#                     # Apply GaussianBlur and ColorJitter only to the image
#                     img = aug(img)
#                 else:
#                     # Apply the same transformation to both img and mask
#                     img = aug(img)
#                     mask = aug(mask)

#         if self.transform:
#             img = self.transform(img)
#             mask = self.transform(mask)
#         return img, mask, img_path

#     def __len__(self):
#         return len(self.imgs)
'''
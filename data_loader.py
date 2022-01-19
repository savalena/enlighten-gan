import torch
from torch.utils.data import Dataset
import os
from utils import image_loader, image_tf


class ABDataset(Dataset):
    def __init__(self, dataset_root, attention_map, phase='train', random_crop_size=320, train=True):
        self.dataset_root = dataset_root
        self.dark_img_dir = os.path.join(dataset_root, phase, 'dark')
        self.normal_img_dir = os.path.join(dataset_root, phase, 'normal')

        self.dark_imgs_filenames = os.listdir(self.dark_img_dir)
        self.normal_imgs_filenames = os.listdir(self.normal_img_dir)

        self.transforms = image_tf.get_transforms(train, random_crop_size)
        self.attention_map = attention_map

    def __getitem__(self, idx):
        dark_img_path = os.path.join(self.dark_img_dir, self.dark_imgs_filenames[idx % len(self.dark_imgs_filenames)])
        normal_img_path = os.path.join(self.normal_img_dir,
                                       self.normal_imgs_filenames[idx % len(self.normal_imgs_filenames)])
        dark_img_raw = image_loader.load_image(dark_img_path)
        norm_img_raw = image_loader.load_image(normal_img_path)

        dark_img = self.transforms(dark_img_raw)
        norm_img = self.transforms(norm_img_raw)

        gray = self.attention_map(dark_img)

        return {'dark': dark_img, 'normal': norm_img, "gray": gray}

    def __len__(self):
        return len(self.dark_imgs_filenames)



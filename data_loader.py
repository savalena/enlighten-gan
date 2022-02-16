import random

import torch
from torch.utils.data import Dataset
import os
from utils import image_utils, image_tf


class ABDataset(Dataset):
    def __init__(self, dataset_root, attention_map, subroot_dark='dark', subroot_normal='normal', phase='train',
                 random_crop_size=320, train=True):
        self.dataset_root = dataset_root
        self.dataset_subroot_dark = subroot_dark
        self.dataset_subroot_normal = subroot_normal
        self.dark_img_dir = os.path.join(dataset_root, phase, self.dataset_subroot_dark)
        self.normal_img_dir = os.path.join(dataset_root, phase, self.dataset_subroot_normal)

        self.dark_imgs_filenames = os.listdir(self.dark_img_dir)
        self.normal_imgs_filenames = os.listdir(self.normal_img_dir)

        self.transforms = image_tf.get_transforms(train, random_crop_size)
        self.attention_map = attention_map

    def __getitem__(self, idx):
        dark_img_path = os.path.join(self.dark_img_dir, self.dark_imgs_filenames[idx % len(self.dark_imgs_filenames)])
        normal_img_path = os.path.join(self.normal_img_dir,
                                       self.normal_imgs_filenames[idx % len(self.normal_imgs_filenames)])
        dark_img_raw = image_utils.load_image(dark_img_path)
        norm_img_raw = image_utils.load_image(normal_img_path)

        dark_img = self.transforms(dark_img_raw)
        norm_img = self.transforms(norm_img_raw)

        if random.random() > 0.5:
            idx = [i for i in range(dark_img.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            dark_img = dark_img.index_select(2, idx)
            norm_img = norm_img.index_select(2, idx)

        if random.random() > 0.5:
            idx = [i for i in range(dark_img.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            dark_img = dark_img.index_select(1, idx)
            norm_img = norm_img.index_select(1, idx)

        if random.random() > 0.5:
            times = random.randint(200, 400) / 100.
            input_img = (dark_img + 1) / 2. / times
            input_img = input_img * 2 - 1
        else:
            input_img = dark_img

        gray = self.attention_map(dark_img)

        return {'dark': dark_img, 'input': input_img, 'normal': norm_img, "gray": gray}

    def __len__(self):
        return len(self.dark_imgs_filenames)

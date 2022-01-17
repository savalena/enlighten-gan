from torch.utils.data import Dataset
import os
from utils import image_loader, image_tf


class ABDatasetLoader(Dataset):
    def __init__(self, dataset_root, phase='train', random_crop_size=256, train=True):
        self.dataset_root = dataset_root
        self.dark_img_dir = os.path.join(dataset_root, phase, 'Dark')
        self.normal_img_dir = os.path.join(dataset_root, phase, 'Norm')

        self.dark_imgs_filenames = os.listdir(self.dark_img_dir)
        self.normal_imgs_filenames = os.listdir(self.normal_img_dir)

        self.transforms = image_tf.get_transforms(train, random_crop_size)

    def __getitem__(self, idx):
        dark_img_path = os.path.join(self.dark_img_dir, self.dark_imgs_filenames[idx % len(self.dark_imgs_filenames)])
        normal_img_path = os.path.join(self.normal_img_dir,
                                       self.normal_imgs_filenames[idx % len(self.normal_imgs_filenames)])
        dark_img_raw = image_loader.load_image(dark_img_path)
        norm_img_raw = image_loader.load_image(normal_img_path)

        dark_img = self.transforms(dark_img_raw)
        norm_img = self.transforms(norm_img_raw)

        return {'dark': dark_img, 'norm': norm_img}

    def __len__(self):
        return len(self.dark_imgs_filenames)

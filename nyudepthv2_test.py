import os
import os.path
import random

import numpy as np
import h5py
from PIL import Image

import torch
from torchvision import transforms 
import torchvision.transforms.functional
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split

iheight, iwidth = 480, 640

class NYUDepthRGBD(Dataset):
    def __init__(self, image_dir, depth_dir, train=True):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.train = train
        self.output_size = (224, 224)
        
        self.ids = sorted([
            fname.replace("rgb_image_", "").replace(".jpg", "")
            for fname in os.listdir(image_dir) if fname.endswith(".jpg")
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        rgb_path = os.path.join(self.image_dir, f"rgb_image_{id_}.jpg")
        depth_path = os.path.join(self.depth_dir, f"depth_map_{id_}.png")

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        depth = np.array(Image.open(depth_path)).astype(np.float32)

        if self.train:
            rgb_tensor, depth_tensor = self.train_transform(rgb, depth)
        else:
            rgb_tensor, depth_tensor = self.val_transform(rgb, depth)

        return rgb_tensor, depth_tensor

    def train_transform(self, rgb, depth):

        # why?
        s = np.random.uniform(1.0, 1.5)
        depth = depth / s

        rgb_pil = Image.fromarray(rgb.copy())
        depth_pil = Image.fromarray(depth.copy())

        # !
        resize1 = transforms.Resize((250, int(250 * 640 / 480)))
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        angle = transforms.RandomRotation.get_params((-5, 5))
        rgb_pil = transforms.functional.rotate(rgb_pil, angle)
        depth_pil = transforms.functional.rotate(depth_pil, angle)

        # !
        dim2 = (int(s * 250), int(s * 250 * 640 / 480))
        resize2 = transforms.Resize(dim2)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)

        center_crop = transforms.CenterCrop((228, 304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        if random.random() > 0.5:
            rgb_pil = transforms.functional.hflip(rgb_pil)
            depth_pil = transforms.functional.hflip(depth_pil)

        resize3 = transforms.Resize(self.output_size)
        rgb_pil = resize3(rgb_pil)
        depth_pil = resize3(depth_pil)

        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_pil = color_jitter(rgb_pil)

        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        rgb_pil = Image.fromarray(rgb)
        depth_pil = Image.fromarray(depth)

        resize1 = transforms.Resize((250, int(250 * 640 / 480)))
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        center_crop = transforms.CenterCrop((228, 304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        resize2 = transforms.Resize(self.output_size)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)

        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))

        return rgb_tensor, depth_tensor


def create_data_loaders():
    train_dataset = NYUDepthRGBD(
        image_dir='dataset_ours/train/images',
        depth_dir='dataset_ours/train/labels',
        train=True
    )

    val_dataset = NYUDepthRGBD(
        image_dir='dataset_ours/val/images',
        depth_dir='dataset_ours/val/labels',
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader

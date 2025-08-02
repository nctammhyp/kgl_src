import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

# Kích thước gốc ảnh NYU
iheight, iwidth = 480, 640

# ===================== Dataset =====================
class DepthDataset(Dataset):
    def __init__(self, paths, mode="train", size=(224, 224)):
        """
        paths: list các tuple (rgb_path, depth_path)
        mode: "train" hoặc "val"
        size: output size (w, h)
        """
        self.paths = paths
        self.mode = mode
        self.output_size = size

        if mode == "train":
            self.transform = self.train_transform
        else:
            self.transform = self.val_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb_path, depth_path = self.paths[index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb_tensor, depth_tensor = self.transform(rgb, depth)
        return rgb_tensor, depth_tensor

    # ==================== AUGMENTATION ====================
    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)
        depth_np = depth / s

        rgb_pil = Image.fromarray(rgb.copy())
        depth_pil = Image.fromarray(depth_np.copy())

        # resize1
        dim1 = (int(250 * 480 / iheight), int(250 * 640 / iheight))
        resize1 = transforms.Resize(dim1)
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        # Random rotation
        angle = transforms.RandomRotation.get_params((-5, 5))
        rgb_pil = TF.rotate(rgb_pil, angle)
        depth_pil = TF.rotate(depth_pil, angle)

        # resize2
        dim2 = (int(s * 250 * 480 / iheight), int(s * 250 * 640 / iheight))
        resize2 = transforms.Resize(dim2)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)

        # Center Crop
        center_crop = transforms.CenterCrop((228, 304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        # Random horizonal flip
        if random.random() > 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            depth_pil = TF.hflip(depth_pil)

        # resize3
        resize3 = transforms.Resize(self.output_size)
        rgb_pil = resize3(rgb_pil)
        depth_pil = resize3(depth_pil)

        # Color Jitter
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
        rgb_pil = color_jitter(rgb_pil)

        # To Tensor
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))

        return rgb_tensor, depth_tensor

    def val_transform(self, rgb, depth):
        rgb_pil = Image.fromarray(rgb)
        depth_pil = Image.fromarray(depth)

        # resize1
        dim1 = (int(250 * 480 / iheight), int(250 * 640 / iheight))
        resize1 = transforms.Resize(dim1)
        rgb_pil = resize1(rgb_pil)
        depth_pil = resize1(depth_pil)

        # Center Crop
        center_crop = transforms.CenterCrop((228, 304))
        rgb_pil = center_crop(rgb_pil)
        depth_pil = center_crop(depth_pil)

        # resize2
        resize2 = transforms.Resize(self.output_size)
        rgb_pil = resize2(rgb_pil)
        depth_pil = resize2(depth_pil)

        # To Tensor
        to_tensor = transforms.ToTensor()
        rgb_tensor = to_tensor(np.array(rgb_pil))
        depth_tensor = to_tensor(np.array(depth_pil))

        return rgb_tensor, depth_tensor


# ===================== Lấy danh sách file ảnh/label =====================
def get_image_label_pairs(directory, img_ext=None, label_ext=None):
    img_dir = os.path.join(directory, "images")
    lbl_dir = os.path.join(directory, "labels")

    img_files = sorted(os.listdir(img_dir))
    lbl_files = sorted(os.listdir(lbl_dir))

    if img_ext:
        img_files = [f for f in img_files if f.endswith(img_ext)]
    if label_ext:
        lbl_files = [f for f in lbl_files if f.endswith(label_ext)]

    files = sorted(set(img_files) & set(lbl_files))
    return [(os.path.join(img_dir, f), os.path.join(lbl_dir, f)) for f in files]


# ===================== Hàm tạo DataLoader =====================
def create_data_loaders(data_root, batch_size=8, size=(224, 224)):
    # Lấy danh sách file train/val
    train_paths = get_image_label_pairs(os.path.join(data_root, "train"), img_ext=".png", label_ext=".png")
    val_paths   = get_image_label_pairs(os.path.join(data_root, "val"), img_ext=".png", label_ext=".png")

    # Shuffle train
    random.shuffle(train_paths)

    # Dataset
    train_dataset = DepthDataset(train_paths, mode="train", size=size)
    val_dataset   = DepthDataset(val_paths, mode="val", size=size)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)

    return train_loader, val_loader


# ===================== Example =====================
if __name__ == "__main__":
    data_root = "/kaggle/input/danything-dataset/danything/danything"
    train_loader, val_loader = create_data_loaders(data_root, batch_size=8, size=(224, 224))

    for rgb, depth in train_loader:
        print(rgb.shape, depth.shape)
        break

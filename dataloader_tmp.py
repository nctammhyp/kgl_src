import os
import cv2
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import albumentations as A
import matplotlib.pyplot as plt


from transform import Resize, NormalizeImage, PrepareForNet, Crop  # giữ nguyên như code Depth-Anything-V2

def get_all_files(directory, rgb_prefix="rgb_image_", depth_prefix="depth_map_"):
    """
    Tạo list chứa tuple (rgb_path, depth_path).
    Dùng khi ảnh RGB và depth được lưu thành file .jpg/.png riêng.
    """
    rgb_files = sorted([f for f in os.listdir(directory + "/images") if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(directory + "/labels") if f.endswith(".png")])

    all_paths = []
    for rgb, depth in zip(rgb_files, depth_files):
        rgb_path = os.path.join(directory, "images", rgb)
        depth_path = os.path.join(directory, "labels", depth)
        all_paths.append((rgb_path, depth_path))
    return all_paths


class NYUDataset(Dataset):
    def __init__(self, paths, mode="train", size=(224, 224)):
        """
        paths: list các tuple (rgb_path, depth_path)
        mode : "train" hoặc "val"
        size : kích thước resize cuối
        """
        self.paths = paths
        self.mode = mode
        self.size = size

        net_w, net_h = size
        # Transform giống Depth-Anything
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == "train" else []))

        # Augmentation riêng cho train
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(hue=0.1, contrast=0.1, brightness=0.1, saturation=0.1, p=0.5),
            A.GaussNoise(
            std_range=[0.03, 0.07],
            mean_range=[0, 0.2],
            per_channel=True,
            noise_scale_factor=1,
            p = 0.3
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb_path, depth_path = self.paths[index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if self.mode == "train":
            augmented = self.augs(image=rgb, mask=depth)
            rgb = augmented["image"] / 255.0
            depth = augmented["mask"]
        else:
            rgb = rgb / 255.0

        sample = self.transform({"image": rgb, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])

        return sample


def get_dataloaders(batch_size):
    train_paths = get_all_files("dataset_ours/nyudepthv2/train")
    val_paths = get_all_files("dataset_ours/nyudepthv2/val")

    train_dataset = NYUDataset(train_paths, mode="train", size=(224, 224))
    val_dataset = NYUDataset(val_paths, mode="val", size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=False)

    return train_loader, val_loader
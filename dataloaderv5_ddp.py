import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


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

        self.resize = transforms.Resize(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize_rgb = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb_path, depth_path = self.paths[index]

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb_pil = Image.fromarray(rgb)
        depth_pil = Image.fromarray(depth)

        rgb_pil = self.resize(rgb_pil)
        depth_pil = self.resize(depth_pil)

        if self.mode == "train" and random.random() > 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            depth_pil = TF.hflip(depth_pil)

        rgb_tensor = self.to_tensor(rgb_pil)
        rgb_tensor = self.normalize_rgb(rgb_tensor)

        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)

        return rgb_tensor, depth_tensor


def get_image_label_pairs(directory, img_ext=".png", label_ext=".png"):
    img_root = os.path.join(directory, "images")
    lbl_root = os.path.join(directory, "labels")

    pairs = []
    if not os.path.exists(img_root) or not os.path.exists(lbl_root):
        return pairs

    for scene_name in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, scene_name)
        lbl_dir = os.path.join(lbl_root, scene_name)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        img_files = set(f for f in os.listdir(img_dir) if f.endswith(img_ext))
        lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(label_ext))

        common_files = sorted(img_files & lbl_files)

        for f in common_files:
            rgb_path = os.path.join(img_dir, f)
            depth_path = os.path.join(lbl_dir, f)
            if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                pairs.append((rgb_path, depth_path))

    return pairs

import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import albumentations as A

from transform import Resize, NormalizeImage, PrepareForNet, Crop  # Giữ nguyên

# =========================
# LẤY FILE RGB + DEPTH
# =========================
def get_all_files_rgbd(directory):
    """
    Tạo list tuple (rgb_path, depth_path) cho dataset_ours/rgbd-scenes-v2
    """
    rgb_files = sorted([f for f in os.listdir(os.path.join(directory, "images")) if f.endswith("-color.png")])
    depth_files = sorted([f for f in os.listdir(os.path.join(directory, "labels")) if f.endswith("-depth.png")])

    all_paths = []
    for rgb, depth in zip(rgb_files, depth_files):
        rgb_path = os.path.join(directory, "images", rgb)
        depth_path = os.path.join(directory, "labels", depth)
        all_paths.append((rgb_path, depth_path))
    return all_paths

def get_all_files_nyu(directory):
    """
    Tạo list tuple (rgb_path, depth_path) cho dataset NYU (giống code cũ)
    """
    rgb_files = sorted([f for f in os.listdir(os.path.join(directory, "images")) if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(os.path.join(directory, "labels")) if f.endswith(".png")])

    all_paths = []
    for rgb, depth in zip(rgb_files, depth_files):
        rgb_path = os.path.join(directory, "images", rgb)
        depth_path = os.path.join(directory, "labels", depth)
        all_paths.append((rgb_path, depth_path))
    return all_paths

# =========================
# LẤY FILE (ảnh, label) TỪ BẤT KỲ DATASET NÀO
# =========================
def get_image_label_pairs(directory, img_ext=None, label_ext=None):
    """
    directory: chứa 2 folder images/labels
    img_ext, label_ext: có thể filter theo phần mở rộng (vd: ".jpg", ".png")
    """
    img_dir = os.path.join(directory, "images")
    lbl_dir = os.path.join(directory, "labels")

    img_files = sorted(os.listdir(img_dir))
    lbl_files = sorted(os.listdir(lbl_dir))

    # Nếu có filter extension
    if img_ext:
        img_files = [f for f in img_files if f.endswith(img_ext)]
    if label_ext:
        lbl_files = [f for f in lbl_files if f.endswith(label_ext)]

    # Match file theo tên (nếu đảm bảo ảnh và label cùng tên)
    files = sorted(set(img_files) & set(lbl_files))

    pairs = [(os.path.join(img_dir, f), os.path.join(lbl_dir, f)) for f in files]
    return pairs


# =========================
# DATASET CHUNG
# =========================
class DepthDataset(Dataset):
    def __init__(self, paths, mode="train", size=(224, 224)):
        self.paths = paths
        self.mode = mode
        self.size = size

        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w, height=net_h,
                # resize_target=True,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14, resize_method="lower_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if mode == "train" else []))

        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(hue=0.1, contrast=0.1, brightness=0.1,
                          saturation=0.1, p=0.5),
            A.GaussNoise(
                std_range=[0.03, 0.07],
                mean_range=[0, 0.2],
                per_channel=True,
                noise_scale_factor=1,
                p=0.3
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
            rgb, depth = augmented["image"] / 255.0, augmented["mask"]
        else:
            rgb = rgb / 255.0

        sample = self.transform({"image": rgb, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        return sample


# =========================
# LẤY DATALOADER CHO 2 DATASET
# =========================
def get_combined_dataloaders(batch_size=16):
    # NYU Depth v2
    # nyu_train = get_all_files_nyu("dataset_ours/nyudepthv2/train")
    # nyu_val = get_all_files_nyu("dataset_ours/nyudepthv2/val")
    # nyu_test = get_all_files_nyu("dataset_ours/nyudepthv2/test")

    # RGBD Scenes v2
    # rgbd_train = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/train")
    # rgbd_val = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/val")
    # rgbd_test = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/test")

    dan_train = get_image_label_pairs("/kaggle/input/danything-dataset/danything/danything/train")
    dan_val   = get_image_label_pairs("/kaggle/input/danything-dataset/danything/danything/val")

    # Gộp 2 dataset lại
    train_paths = dan_train
    val_paths = dan_val

    # Shuffle train
    random.shuffle(train_paths)

    train_dataset = DepthDataset(train_paths, mode="train", size=(224, 224))
    val_dataset = DepthDataset(val_paths, mode="val", size=(224, 224))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)
    return train_loader, val_loader
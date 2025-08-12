import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


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

        # Transform chung cho RGB
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

        # Đọc ảnh
        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb_pil = Image.fromarray(rgb)
        depth_pil = Image.fromarray(depth)

        # Resize
        rgb_pil = self.resize(rgb_pil)
        depth_pil = self.resize(depth_pil)

        # Flip ngang (train)
        if self.mode == "train" and random.random() > 0.5:
            rgb_pil = TF.hflip(rgb_pil)
            depth_pil = TF.hflip(depth_pil)

        # RGB -> Tensor + Normalize (ImageNet)
        rgb_tensor = self.to_tensor(rgb_pil)
        rgb_tensor = self.normalize_rgb(rgb_tensor)

        # Depth -> Tensor + scale về [0,1] (chia 255)
        depth_np = np.array(depth_pil, dtype=np.float32)
        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # (1,H,W)

        return rgb_tensor, depth_tensor


# ===================== Lấy danh sách file ảnh/label =====================
def get_image_label_pairs(directory, img_ext=".png", label_ext=".png"):
    img_root = os.path.join(directory, "images")
    lbl_root = os.path.join(directory, "labels")

    pairs = []
    if not os.path.exists(img_root) or not os.path.exists(lbl_root):
        return pairs  # Không có thư mục thì trả rỗng

    for scene_name in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, scene_name)
        lbl_dir = os.path.join(lbl_root, scene_name)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        # Lọc file ảnh và depth
        img_files = set(f for f in os.listdir(img_dir) if f.endswith(img_ext))
        lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(label_ext))

        # Lấy giao nhau để chắc chắn có cả RGB và depth
        common_files = sorted(img_files & lbl_files)

        for f in common_files:
            rgb_path = os.path.join(img_dir, f)
            depth_path = os.path.join(lbl_dir, f)
            if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                pairs.append((rgb_path, depth_path))

    return pairs


def subdataset_get_image_label_pairs(directory, img_ext=".png", label_ext=".png", phase = "train"):
    img_root = os.path.join(directory, "images")
    lbl_root = os.path.join(directory, "labels")

    pairs = []
    if not os.path.exists(img_root) or not os.path.exists(lbl_root):
        return pairs  # Không có thư mục thì trả rỗng

    # Danh sách folder được phép lấy
    allowed_scenes = {
        # "scene_1_3MP",
        # "scene_1_12MP",
        "scene_2_12MP",
        "scene_3_12MP",
        "scene_4_12MP",
        "scene_5_12MP",
        "scene_6_12MP",
        "scene_7_12MP"
    }

    for scene_name in sorted(os.listdir(img_root)):
        if phase == 'train' and scene_name not in allowed_scenes:
            continue  # Bỏ qua folder không nằm trong danh sách

        img_dir = os.path.join(img_root, scene_name)
        lbl_dir = os.path.join(lbl_root, scene_name)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        # Lọc file ảnh và depth
        img_files = set(f for f in os.listdir(img_dir) if f.endswith(img_ext))
        lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(label_ext))

        # Lấy giao nhau để chắc chắn có cả RGB và depth
        common_files = sorted(img_files & lbl_files)

        for f in common_files:
            rgb_path = os.path.join(img_dir, f)
            depth_path = os.path.join(lbl_dir, f)
            if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                pairs.append((rgb_path, depth_path))

    return pairs



# ===================== Hàm tạo DataLoader =====================
def create_data_loaders(data_root, batch_size=64, size=(160, 128)):
    # train_paths = get_image_label_pairs(os.path.join(data_root, "train"))
    # val_paths   = get_image_label_pairs(os.path.join(data_root, "val"))


    train_paths = subdataset_get_image_label_pairs(os.path.join(data_root, "train"), phase = "train")
    val_paths   = subdataset_get_image_label_pairs(os.path.join(data_root, "val"), phase = "val")


    random.shuffle(train_paths)

    train_dataset = DepthDataset(train_paths, mode="train", size=size)
    val_dataset   = DepthDataset(val_paths, mode="val", size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4,
                            pin_memory=True)

    return train_loader, val_loader

    # train_paths = get_image_label_pairs(os.path.join(data_root, "train"))
    # val_paths   = get_image_label_pairs(os.path.join(data_root, "val"))

    # # Tạo dãy 3 số ngẫu nhiên từ 1 đến 100
    # random_numbers = random.sample(range(1, 101), 3)
    # print("Random index:", random_numbers)

    # # Lấy 1/5 phần ngẫu nhiên
    # train_paths = random.sample(train_paths, len(train_paths)//5)
    # val_paths = random.sample(val_paths, len(val_paths)//5)

    # random.shuffle(train_paths)

    # train_dataset = DepthDataset(train_paths, mode="train", size=size)
    # val_dataset   = DepthDataset(val_paths, mode="val", size=size)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                         shuffle=True, num_workers=4,
    #                         pin_memory=True, drop_last=True)

    # val_loader = DataLoader(val_dataset, batch_size=1,
    #                         shuffle=False, num_workers=4,
    #                         pin_memory=True)

    # return train_loader, val_loader



# ===================== Example =====================
if __name__ == "__main__":
    data_root = "/path/to/dataset"
    train_loader, val_loader = create_data_loaders(data_root, batch_size=8, size=(224, 224))

    for rgb, depth in train_loader:
        print(rgb.shape, depth.shape, rgb.min().item(), rgb.max().item(), depth.min().item(), depth.max().item())
        break
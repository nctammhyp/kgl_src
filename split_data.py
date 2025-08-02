import os
import shutil
from glob import glob
import random

# Đường dẫn gốc
rgb_dir = 'nyu_rgb_images'
depth_dir = 'nyu_depth_images'

# Danh sách file
rgb_files = sorted(glob(os.path.join(rgb_dir, 'rgb_image_*.jpg')))
depth_files = sorted(glob(os.path.join(depth_dir, 'depth_map_*.png')))

# Kiểm tra số lượng khớp nhau
assert len(rgb_files) == len(depth_files), "Số lượng RGB và Depth không khớp!"

# Shuffle để ngẫu nhiên hóa
combined = list(zip(rgb_files, depth_files))
random.seed(42)
random.shuffle(combined)

# Chia tỷ lệ
total = len(combined)
train_ratio = 0.8
val_ratio = 0.1

n_train = int(train_ratio * total)
n_val = int(val_ratio * total)

splits = {
    'train': combined[:n_train],
    'val': combined[n_train:n_train+n_val],
    'test': combined[n_train+n_val:]
}

# Hàm tạo thư mục
def make_dirs(base_path):
    os.makedirs(os.path.join(base_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'labels'), exist_ok=True)

# Di chuyển file
output_root = 'dataset_ours'
for split, data in splits.items():
    make_dirs(os.path.join(output_root, split))
    for rgb_path, depth_path in data:
        filename = os.path.basename(rgb_path)
        depthname = os.path.basename(depth_path)

        shutil.copy(rgb_path, os.path.join(output_root, split, 'images', filename))
        shutil.copy(depth_path, os.path.join(output_root, split, 'labels', depthname))

print("✅ Done splitting into train / val / test.")

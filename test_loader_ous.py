import os

from nyudepthv2_test import create_data_loaders, NYUDepthRGBD
import utils, loss_func
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import numpy as np



train_loader, val_loader = create_data_loaders()

val_dataset = NYUDepthRGBD(
        image_dir='dataset_ours/val/images',
        depth_dir='dataset_ours/val/labels',
        train=False
    )

plt.close()

# images, depths = val_dataset[20]
# image = images[0]  # C:3, H, W
# depth = depths[0]  # 1 hoặc HxW

# print(f"image shape: {image.shape}")
# print(f"depth shape: {depth.shape}")

# # Chuyển tensor về numpy để hiển thị
# image_np = TF.to_pil_image(image)
# depth_np = depth.squeeze().cpu().numpy()

# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.title("RGB Image")
# plt.imshow(image_np)
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Depth Map")
# plt.imshow(depth_np, cmap='plasma')
# plt.axis('off')
# plt.show()


# print(val_loader)

# Lấy 1 batch từ val_loader
for batch in val_loader:
    images, depths = batch  # Giả sử val_loader trả về tuple (images, depths)

    print("Image batch shape:", images.shape)
    print("Depth batch shape:", depths.shape)

    # Hiển thị ảnh đầu tiên và bản đồ độ sâu tương ứng
    image = images[0]  # C:3, H, W
    depth = depths[0]  # 1 hoặc HxW

    print(f"image shape: {image.shape}")
    print(f"depth shape: {depth.shape}")

    # Chuyển tensor về numpy để hiển thị
    image_np = TF.to_pil_image(image)
    depth_np = depth.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("RGB Image")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth_np, cmap='plasma')
    plt.axis('off')

    print("Active figures:", plt.get_fignums())


    plt.show()
    plt.close()  # tránh hiện lại ở lần chạy sau


    break  # chỉ hiển thị 1 batch

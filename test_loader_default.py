import os

from nyudepthv2 import create_data_loaders, data_loader_test
import utils, loss_func
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

global args, writer



args = utils.parse_args()



val_loader = data_loader_test(args)

plt.close()


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

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from model_v4 import FastDepthV2  # giữ nguyên

# ================== Hàm load ảnh ==================
iheight, iwidth = 480, 640

def load_test_image(rgb, size=(224,224)):
    rgb_pil = Image.fromarray(rgb)

    dim1 = (int(250 * 480 / iheight), int(250 * 640 / iheight))
    rgb_pil = transforms.Resize(dim1)(rgb_pil)
    rgb_pil = transforms.CenterCrop((228, 304))(rgb_pil)
    rgb_pil = transforms.Resize(size)(rgb_pil)

    return transforms.ToTensor()(np.array(rgb_pil)).unsqueeze(0)

def load_image(rgb_path):
    img = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_test_image(rgb), rgb

def load_pretrained_fastdepth(model, weights_path):
    assert os.path.isfile(weights_path), f"Không tìm thấy {weights_path}"
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model

# ================== Hàm inference ==================
def visualize_folder_images(folder_path, model_path, use_cuda=True):
    # Load model
    model = FastDepthV2()
    model = load_pretrained_fastdepth(model, model_path)
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Lấy danh sách ảnh
    img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith((".png",".jpg",".jpeg"))])
    if not img_files:
        print("❌ Không có ảnh trong thư mục:", folder_path)
        return

    for fname in img_files:
        img_path = os.path.join(folder_path, fname)
        img_tensor, img_rgb = load_image(img_path)
        if use_cuda and torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            pred_depth = model(img_tensor)

        pred_depth = pred_depth.squeeze().cpu().numpy()
        pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())

        # Hiển thị
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title(f"Input: {fname}")
        plt.imshow(img_rgb)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title("Predicted Depth")
        plt.imshow(pred_norm, cmap='viridis')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ================== Chạy thử ==================
if __name__ == "__main__":
    dir_img_test = "/kaggle/input/danything-dataset/FastDepth_src/FastDepth_src/test_unlabel/"
    model_path = "/kaggle/working/ours_checkpoints/checkpoint_best_387.pth"
    visualize_folder_images(dir_img_test, model_path, use_cuda=True)

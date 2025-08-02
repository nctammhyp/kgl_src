import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from models import FastDepth
import matplotlib.pyplot as plt
import os

from transform import Resize, NormalizeImage, PrepareForNet, Crop  # Giữ nguyên
from torchvision.transforms import Compose


def load_image(rgb_path, device="cuda"):
    img = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     # Tạo transform giống như dataset (không augment)
    net_w, net_h = 224, 224
    transform = Compose([
        Resize(width=net_w, height=net_h,
               resize_target=False,
               keep_aspect_ratio=True,
               ensure_multiple_of=14,
               resize_method="lower_bound",
               image_interpolation_method=cv2.INTER_CUBIC),
        NormalizeImage(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        PrepareForNet()
    ])

    # Chuẩn bị input
    sample = transform({"image": rgb})
    img_tensor = torch.from_numpy(sample["image"]).unsqueeze(0).to(device)  # (1,C,H,W)

    return img_tensor, rgb

def load_pretrained_fastdepth(model,weights_path):
        assert os.path.isfile(weights_path), "No pretrained model found. abort.."
        print('Model found, loading...')
        # checkpoint = torch.load(weights_path)
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.load_state_dict(checkpoint['model'])
        print('Finished loading')
        return model

def inference(image_path, model_path, use_cuda=True):
    # Load model
    model = FastDepth()
    model, _ = load_pretrained_fastdepth(model, model_path)

    if use_cuda and torch.cuda.is_available():
        model.cuda()
    model.eval()

    # Load and preprocess image
    img_tensor, img_rgb = load_image(image_path)
    if use_cuda and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    # Inference
    with torch.no_grad():
        pred_depth = model(img_tensor)

    # Convert sang numpy
    pred_depth = pred_depth.squeeze().cpu().numpy()

    # Chuẩn hóa về 0-1 để hiển thị
    pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)



    # Show image and prediction
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth")
    plt.imshow(pred_norm, cmap='inferno')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = ""
    model_path = ""
    inference(image_path, model_path, use_cuda=True)

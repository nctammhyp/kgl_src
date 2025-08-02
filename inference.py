import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
from models import FastDepth
from load_pretrained import load_pretrained_fastdepth
import matplotlib.pyplot as plt

def load_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, size)
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)  # Shape: [1, 3, H, W]
    return img_tensor, img_resized

def normalize_depth(depth):
    depth_np = depth.squeeze().cpu().numpy()
    depth_np -= depth_np.min()
    depth_np /= (depth_np.max() + 1e-8)
    return depth_np

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
        pred = model(img_tensor)

    depth = normalize_depth(pred)

    # Show image and prediction
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Depth")
    plt.imshow(depth, cmap='inferno')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, help='Path to input image', default='data_test/test.png')
    parser.add_argument('--model', type=str, required=False, help='Path to .pth model', default='Weights/FastDepth_L1_Best.pth')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available', default=False)
    args = parser.parse_args()

    inference(args.image, args.model, use_cuda=args.gpu)

import cv2, torch, torchvision.transforms as transforms
import numpy as np, os
from PIL import Image
from model_v4 import FastDepthV2

iheight, iwidth = 480, 640

def load_test_image(rgb, size=(224,224)):
    rgb_pil = Image.fromarray(rgb)
    rgb_pil = transforms.Resize((int(250 * 480 / iheight), int(250 * 640 / iheight)))(rgb_pil)
    rgb_pil = transforms.CenterCrop((228, 304))(rgb_pil)
    rgb_pil = transforms.Resize(size)(rgb_pil)
    return transforms.ToTensor()(np.array(rgb_pil)).unsqueeze(0)

def load_image(rgb_path):
    img = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return load_test_image(rgb), rgb

def load_pretrained_fastdepth(model, weights_path):
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)  # Vì bạn đã lưu trực tiếp state_dict
    return model


def inference(image_path, model_path, use_cuda=True):
    model = FastDepthV2()
    model = load_pretrained_fastdepth(model, model_path)
    if use_cuda and torch.cuda.is_available():
        model.cuda()
    model.eval()

    img_tensor, img_rgb = load_image(image_path)
    if use_cuda and torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    with torch.no_grad():
        pred_depth = model(img_tensor)

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min())
    return img_rgb, pred_norm

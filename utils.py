"""
python3 main.py -mode train
"""

import collections
import argparse
import torchvision
from torchvision import transforms
import torch
import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt
import cv2
plt.set_cmap("jet")
def parse_args():
    loss_functions = ['l1','l2','rmsle','l1gn','l2gn','rmslegn']
    backbone_model = ['mobilenet','mobilenetv2']
    parser = argparse.ArgumentParser(description='FastDepth')
    parser.add_argument('-mode', help='High level recipe: write tensors, train, test or evaluate models.')
    parser.add_argument('--backbone',default='mobilenetv2',type=str,help=f'Which backbone to use, options are: {backbone_model} (default is mobilenet)')
    parser.add_argument('--weights_dir',default='Weights',type=str,help='Directory to save and load trained weights.')
    args = parser.parse_args()

    print('Arguments are', args)

    return args


def save_best_samples(imgs_dict):
    tensors_list = []
    sorted_imgs = collections.OrderedDict(sorted(imgs_dict.items(),reverse=True))
    for value in sorted_imgs:
        tensors_list.append(sorted_imgs[value])  
    rgb,pred,depth = [],[],[]
    for i in tensors_list:
        rgb.append(i[0])
        pred.append(i[1])
        depth.append(i[2])
    
    rgb = torch.stack(rgb).squeeze(1)
    pred = torch.stack(pred).squeeze(1)
    depth = torch.stack(depth).squeeze(1)
    rgb_grid = torchvision.utils.make_grid(rgb[:8],normalize=True)
    pred_grid = torchvision.utils.make_grid(pred[:8],normalize=True)
    depth_grid = torchvision.utils.make_grid(depth[:8],normalize=True)

    samples = torch.stack((rgb_grid,pred_grid,depth_grid))
    samples_grid = torchvision.utils.make_grid(samples,nrow=1,normalize= True)
    data_transforms = transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1)])
    gray_samples = data_transforms(samples_grid.cpu())
    torchvision.utils.save_image(samples_grid,'samples.png')
    matplotlib.image.imsave('samplesjet.png', gray_samples)

import os
import cv2
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import albumentations as A
import matplotlib.pyplot as plt

from model_v4 import FastDepthV2, FastDepth, weights_init
import dataloader_v4
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
import torch.optim as optim


import utils, loss_func
from metric_depth.util.loss import SiLogLoss, DepthLoss
from torch.optim.lr_scheduler import LambdaLR

import math
from tqdm import tqdm
import torch.nn.functional as F
import json

import glob


if __name__ == "__main__":
    model = FastDepthV2().to("cpu")
    model.encoder = load_pretrained_encoder(model.encoder,'Weights','mobilenetv2')
    model.decoder.apply(weights_init)

    # Đặt model ở chế độ train
    model.train()

    # Tạo input giả: batch_size=2, 3 kênh RGB, 224x224
    dummy_input = torch.randn(2, 3, 160, 128)

    # Forward
    output = model(dummy_input)
    print("Output shape:", output.shape)

    # Tạo loss giả để test backward
    dummy_target = torch.randn_like(output)
    criterion = torch.nn.MSELoss()
    loss = criterion(output, dummy_target)
    print("Loss:", loss.item())

    # Backward
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Forward/backward OK ✅")





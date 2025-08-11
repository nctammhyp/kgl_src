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
import dataloader_v5
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


args = utils.parse_args()



def eval_depth(pred, target):
    valid_mask = ((target>0) + (pred>0)) > 0
    pred_output = pred[valid_mask]
    target_masked =  target[valid_mask]
    abs_diff = (pred_output - target_masked).abs() 
    RMSE = torch.sqrt(torch.mean((abs_diff).pow(2)))

    maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
    d1 = float((maxRatio < 1.25).float().mean())

    return {
        'd1': d1,
        'rmse': RMSE.item(),
    }




def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



def train_fn(device = "cpu", load_state = False, state_path = './'):
    # params
    num_epochs = 50


    model = FastDepthV2().to("cuda:0")
    model.encoder = load_pretrained_encoder(model.encoder,args.weights_dir,args.backbone)
    model.decoder.apply(weights_init)

    # optim = torch.optim.SGD(model.parameters(), lr = 3e-4 ,weight_decay=1e-4)
    optim = torch.optim.AdamW(
          model.parameters(),  # lấy toàn bộ parameter của model
          lr=3e-4,
          weight_decay=0.01
      )


    print('Model created')

    # criterion = SiLogLoss() # author's loss
    criterion = DepthLoss()

    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))

    train_loader, val_loader = dataloader_v5.create_data_loaders("/kaggle/input/hypdataset-v1-0/hypdataset_v1", batch_size=512)
 

    best_val_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        checkpoint = torch.load("/kaggle/working/ours_checkpoints/checkpoint_best_358.pth", map_location=device)
        # model.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])

        model.load_state_dict(checkpoint)
        model = model.to("cuda")


    accum_steps = 8  # số batch muốn cộng dồn trước khi update weights

    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0

        optim.zero_grad()

        for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            img, depth = input.to(device), target.to(device)

            pred = model(img)
            loss = criterion('l1', pred, depth, epoch)
            loss = loss / accum_steps   # chia loss để giữ scale ổn định
            loss.backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optim.step()
                optim.zero_grad()

            total_loss += loss.item() * accum_steps  # nhân lại để logging đúng

        avg_loss = total_loss / len(train_loader)


        # ===== Validation =====
        model.eval()
        results = {'d1': 0, 'rmse': 0}
        test_loss = 0

        with torch.no_grad():
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = model(img)

                test_loss += criterion('l1',pred, depth).item()

                # mask = (depth >= 0.001)
                cur_results = eval_depth(pred, depth)


                for k in results:
                    results[k] += cur_results[k]

        
        val_loss = test_loss/len(val_loader)

        # for k in results:
        #    results[k] = round(results[k] / len(val_loader), 3)
        for k in results:
            results[k] = round((results[k] / len(val_loader)), 3)



        # ===== Save Checkpoint =====
        # torch.save({
        #     "model": model.state_dict(),
        #     "optim": optim.state_dict()
        #     # "scheduler": scheduler.state_dict()
        # }, f"{state_path}/checkpoint_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            new_ckpt = f"{state_path}/checkpoint_best_{epoch}.pth"

            # 1. Lưu checkpoint mới
            torch.save(model.state_dict(), new_ckpt)

            # 2. Xóa tất cả checkpoint cũ (trừ file vừa lưu)
            for ckpt in glob.glob(f"{state_path}/checkpoint_best_*.pth"):
                if ckpt != new_ckpt:
                    os.remove(ckpt)

        # Cập nhật history
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        # Lưu log JSON
        with open(f"{state_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)


        print(f"epoch_{epoch}, train_loss={avg_loss:.5f}, val loss: {val_loss:.5f}, val_metrics={results}")

        # ==== Vẽ biểu đồ ====
        # epochs = range(1, num_epochs+1)
        epochs = range(1, len(history["train_loss"])+1)

        plt.figure(figsize=(8,5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{state_path}/train_loss_curve.png")
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{state_path}/val_loss_curve.png")
        plt.close()



if __name__ == "__main__":
    train_fn(device='cuda:0', load_state=False, state_path="/kaggle/working/ours_checkpoints")
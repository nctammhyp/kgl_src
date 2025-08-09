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
    eps = 1e-6  # tránh chia 0, log 0
    assert pred.shape == target.shape

    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    diff = pred_safe - target_safe
    diff_log = torch.log(pred_safe) - torch.log(target_safe)

    abs_rel = torch.mean(torch.abs(diff) / target_safe)
    rmse = torch.sqrt(torch.mean(diff ** 2))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(
        torch.mean(diff_log ** 2) - 0.5 * (torch.mean(diff_log) ** 2)
    )

    return {
        'd1': d1.detach(),
        'abs_rel': abs_rel.detach(),
        'rmse': rmse.detach(),
        'mae': mae.detach(),
        'silog': silog.detach()
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

    optim = torch.optim.AdamW(
          model.parameters(),  # lấy toàn bộ parameter của model
          lr=1e-3,
          weight_decay=0.01
      )

    print('Model created')

    criterion = SiLogLoss() # author's loss
    # criterion = DepthLoss()

    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))

    train_loader, val_loader = dataloader_v5.create_data_loaders("/kaggle/input/danything-dataset/danything/danything", batch_size=64, size=(224, 224))
 

    best_val_absrel = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        checkpoint = torch.load("/kaggle/working/ours_checkpoints/checkpoint_best_358.pth", map_location=device)
        # model.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])

        model.load_state_dict(checkpoint)
        model = model.to("cuda")


    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0

        for i , (input,target) in tqdm(enumerate(train_loader)):
            img, depth = input.to(device), target.to(device)

            optim.zero_grad()
            pred = model(img).squeeze(1)  

            # loss = criterion('l1',pred,depth,epoch)

            mask = (depth > 1e-3) & (depth <= 255) & torch.isfinite(depth)
            loss = criterion(pred, depth, mask)



            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        # results = {'d1': 0, 'rmse': 0}
        results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'silog': 0}
        # test_loss = 0

        with torch.no_grad():
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = model(img)

                # test_loss += criterion('l1',pred, depth).item()
                pred = pred.squeeze(1).squeeze(0)

                # mask = (depth >= 0.001)
                # cur_results = eval_depth(pred, depth)
                mask = (depth <= 255) & (depth >= 0.001)
                cur_results = eval_depth(pred[mask], depth[mask])


                for k in results:
                    results[k] += cur_results[k]

        
        # val_loss = test_loss/len(val_loader)

        # for k in results:
        #    results[k] = round(results[k] / len(val_loader), 3)
        for k in results:
            results[k] = round((results[k] / len(val_loader)).item(), 3)



        # ===== Save Checkpoint =====
        # torch.save({
        #     "model": model.state_dict(),
        #     "optim": optim.state_dict()
        #     # "scheduler": scheduler.state_dict()
        # }, f"{state_path}/checkpoint_{epoch}.pth")

        if results['abs_rel'] < best_val_absrel:
            best_val_absrel = results['abs_rel']
            new_ckpt = f"{state_path}/checkpoint_best_{epoch}.pth"

            # 1. Lưu checkpoint mới
            torch.save(model.state_dict(), new_ckpt)

            # 2. Xóa tất cả checkpoint cũ (trừ file vừa lưu)
            for ckpt in glob.glob(f"{state_path}/checkpoint_best_*.pth"):
                if ckpt != new_ckpt:
                    os.remove(ckpt)

        # Cập nhật history
        history["train_loss"].append(avg_loss)
        # history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        # Lưu log JSON
        with open(f"{state_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)


        print(f"epoch_{epoch}, train_loss={avg_loss:.5f}, val_metrics={results}")

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

        absrel = [m["abs_rel"] for m in history["val_metrics"]]
        plt.figure(figsize=(8,5))
        plt.plot(epochs, absrel, label="AbsRel (val)")
        plt.xlabel("Epoch")
        plt.ylabel("AbsRel")
        plt.legend()
        plt.savefig(f"{state_path}/val_absrel_curve.png")
        plt.close()



if __name__ == "__main__":
    train_fn(device='cuda:0', load_state=True, state_path="/kaggle/working/ours_checkpoints")
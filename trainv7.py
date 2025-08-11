# train.py
import os
import json
import glob
import math
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

from model_v4 import FastDepthV2, weights_init
import dataloader_v5
from load_pretrained import load_pretrained_encoder
from metric_depth.util.loss import DepthLoss
import utils

args = utils.parse_args()

def eval_depth(pred, target):
    valid_mask = ((target > 0) + (pred > 0)) > 0
    pred_output = pred[valid_mask]
    target_masked = target[valid_mask]
    abs_diff = (pred_output - target_masked).abs()
    RMSE = torch.sqrt(torch.mean(abs_diff.pow(2)))
    maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
    d1 = float((maxRatio < 1.25).float().mean())
    return {'d1': d1, 'rmse': RMSE.item()}

def train_fn(load_state=False, state_path='./'):
    accelerator = Accelerator(
        gradient_accumulation_steps=8,
        mixed_precision="fp16"
    )

    num_epochs = 50

    model = FastDepthV2()
    model.encoder = load_pretrained_encoder(model.encoder, args.weights_dir, args.backbone)
    model.decoder.apply(weights_init)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = DepthLoss()

    train_loader, val_loader = dataloader_v5.create_data_loaders(
        "/kaggle/input/hypdataset-v1-0/hypdataset_v1", batch_size=256
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        checkpoint = torch.load(f"{state_path}/checkpoint_best.pth", map_location="cpu")
        model.load_state_dict(checkpoint)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img, depth in tqdm(train_loader, disable=not accelerator.is_local_main_process):
            with accelerator.accumulate(model):
                pred = model(img)
                loss = criterion('l1', pred, depth, epoch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss, results = 0, {'d1': 0, 'rmse': 0}
        with torch.no_grad():
            for img, depth in val_loader:
                pred = model(img)
                loss = criterion('l1', pred, depth)
                val_loss += loss.item()
                cur_results = eval_depth(pred, depth)
                for k in results:
                    results[k] += cur_results[k]

        val_loss /= len(val_loader)
        for k in results:
            results[k] = round(results[k] / len(val_loader), 3)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.save(model.state_dict(), f"{state_path}/checkpoint_best.pth")

        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        if accelerator.is_local_main_process:
            accelerator.print(f"Epoch {epoch}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, metrics={results}")
            with open(f"{state_path}/history.json", "w") as f:
                json.dump(history, f, indent=2)

    if accelerator.is_local_main_process:
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.legend()
        plt.savefig(f"{state_path}/loss_curve.png")

if __name__ == "__main__":
    os.makedirs("/kaggle/working/ours_checkpoints", exist_ok=True)
    train_fn(load_state=False, state_path="/kaggle/working/ours_checkpoints")

import os
import glob
import json
import math
import torch
import random
import numpy as np
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import utils
from model_v4 import FastDepthV2, weights_init
import dataloaderv5_ddp
from load_pretrained import load_pretrained_encoder
from metric_depth.util.loss import DepthLoss

torch.backends.cudnn.benchmark = True


def eval_depth(pred, target):
    valid_mask = ((target > 0) + (pred > 0)) > 0
    pred_output = pred[valid_mask]
    target_masked = target[valid_mask]
    abs_diff = (pred_output - target_masked).abs()
    RMSE = torch.sqrt(torch.mean(abs_diff.pow(2)))
    maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
    d1 = float((maxRatio < 1.25).float().mean())
    return {'d1': d1, 'rmse': RMSE.item()}


def setup_ddp():
    """Khởi tạo process group dựa trên biến môi trường."""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def parse_size(s):
    # Cho phép parse chuỗi "160,128" thành tuple (160, 128)
    if isinstance(s, tuple):
        return s
    return tuple(map(int, s.split(',')))


def train_fn():
    rank, world_size = setup_ddp()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master_process = (rank == 0)

    # parse args từ env hoặc add argparser nếu muốn (ví dụ hardcode tạm)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights_dir", type=str, default="/kaggle/working/kgl_src/Weights")
    parser.add_argument("--backbone", type=str, default="mobilenetv2")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/hypdataset-v1-0/hypdataset_v1")
    parser.add_argument("--state_path", type=str, default="/kaggle/working/ours_checkpoints")
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--size", type=str, default="160,128")
    parser.add_argument("--num_epochs", type=int, default=50)
    args = parser.parse_args()

    args.size = parse_size(args.size)  # chuyển từ "160,128" thành tuple (160,128)

    if master_process:
        print(f"Rank {rank} / World Size {world_size}")
        print(f"Using device: {device}")
        print(f"Batch size per process: {args.batch_size}")
        print(f"Resize size: {args.size}")

    # Load paths
    train_paths = dataloaderv5_ddp.get_image_label_pairs(os.path.join(args.data_root, "train"))
    val_paths = dataloaderv5_ddp.get_image_label_pairs(os.path.join(args.data_root, "val"))

    # Dataset
    train_dataset = dataloaderv5_ddp.DepthDataset(train_paths, mode="train", size=args.size)
    val_dataset = dataloaderv5_ddp.DepthDataset(val_paths, mode="val", size=args.size)

    # Distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model setup
    model = FastDepthV2()
    model.encoder = load_pretrained_encoder(model.encoder, args.weights_dir, args.backbone)
    model.decoder.apply(weights_init)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = DepthLoss()

    scaler = torch.amp.GradScaler()

    best_val_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if args.load_state:
        ckpt_path = os.path.join(args.state_path, "checkpoint_best.pth")
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt)
            if master_process:
                print(f"Loaded checkpoint from {ckpt_path}")
        else:
            if master_process:
                print(f"Checkpoint path {ckpt_path} does not exist")

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch} Train", disable=not master_process)
        for input, target in pbar:
            img = input.to(device, non_blocking=True)
            depth = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                pred = model(img)
                loss = criterion('l1', pred, depth, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if master_process:
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        results = {'d1': 0.0, 'rmse': 0.0}
        test_loss = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Rank {rank} Epoch {epoch} Val", disable=not master_process)
            for input, target in vbar:
                img = input.to(device, non_blocking=True)
                depth = target.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda'):
                    pred = model(img)
                    l = criterion('l1', pred, depth)

                test_loss += l.item()
                cur_results = eval_depth(pred, depth)
                for k in results:
                    results[k] += cur_results[k]

                if master_process:
                    vbar.set_postfix(val_loss=test_loss / (vbar.n + 1))

        # Trung bình metric trên tất cả GPU
        for key in results:
            t = torch.tensor(results[key], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            results[key] = (t / world_size).item()

        # Trung bình loss val (quan trọng để save model)
        t_loss = torch.tensor(test_loss, device=device)
        dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
        val_loss = (t_loss / world_size).item() / len(val_loader)

        if master_process:
            print(f"Epoch {epoch}: train_loss={avg_loss:.5f}, val_loss={val_loss:.5f}, metrics={results}")
            os.makedirs(args.state_path, exist_ok=True)

            # Save checkpoint chỉ khi val_loss tốt hơn
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.state_path, f"checkpoint_best_{epoch}.pth")
                torch.save(model.module.state_dict(), ckpt_path)
                # Xóa checkpoint cũ khác
                for f in glob.glob(os.path.join(args.state_path, "checkpoint_best_*.pth")):
                    if f != ckpt_path:
                        os.remove(f)

            # Lưu lịch sử train
            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(results)
            with open(os.path.join(args.state_path, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    cleanup_ddp()


if __name__ == "__main__":
    train_fn()

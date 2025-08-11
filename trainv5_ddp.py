# train.py (DDP-aware, supports single-GPU run)
import os
import math
import torch
import random
import numpy as np
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from model_v4 import FastDepthV2, weights_init
from metric_depth.util.loss import DepthLoss
import dataloaderv5_ddp
import utils

torch.backends.cudnn.benchmark = True

# -----------------------------
# DDP setup & cleanup
# -----------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

# helper to reduce/aggregate a scalar across ranks (sum)
def dist_reduce_scalar(x, device):
    t = torch.tensor(x, device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()

# -----------------------------
# eval helper: aggregate metrics across ranks
# -----------------------------
def aggregate_metrics_across_ranks(sum_dict, count, device):
    """
    sum_dict: {'loss': float, 'd1': float, ...} sums computed on each rank for its subset
    count: number of samples used by this rank (int)
    returns: averaged metrics over total samples across all ranks
    """
    # pack into tensor: [sum_loss, sum_d1, count]
    sum_loss = float(sum_dict.get("loss", 0.0))
    sum_d1 = float(sum_dict.get("d1", 0.0))
    packed = torch.tensor([sum_loss, sum_d1, float(count)], device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_loss = packed[0].item()
    total_d1 = packed[1].item()
    total_count = int(packed[2].item())
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    avg_d1 = total_d1 / total_count if total_count > 0 else 0.0
    return {"loss": avg_loss, "d1": avg_d1, "count": total_count}

# -----------------------------
# Training function
# -----------------------------
def train(args):
    # Init DDP or single-GPU
    if args.ddp:
        rank, world_size = setup_ddp()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        master_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        master_process = True

    # Decide batch_size per process
    # If user specifies --batch_is_global, we divide by world_size to get per-GPU batch.
    if args.batch_is_global and world_size > 1:
        per_gpu_batch = max(1, args.batch_size // world_size)
        if master_process:
            print(f"[INFO] Interpreting --batch_size as GLOBAL. world_size={world_size}, using per-gpu batch={per_gpu_batch}")
    else:
        per_gpu_batch = args.batch_size

    if master_process:
        print(f"[Rank {rank}] Running on {device}, world_size={world_size}, per_gpu_batch={per_gpu_batch}", flush=True)

    # Seed for reproducibility
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    random.seed(42 + rank)

    # Load dataset (you must implement these loader functions accordingly)
    train_paths = dataloaderv5_ddp.get_image_label_pairs(os.path.join(args.train_dir, "train"))
    val_paths = dataloaderv5_ddp.get_image_label_pairs(os.path.join(args.val_dir, "val"))

    train_dataset = dataloaderv5_ddp.DepthDataset(train_paths, mode="train", size=args.size)
    val_dataset = dataloaderv5_ddp.DepthDataset(val_paths, mode="val", size=args.size)

    if args.ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # For validation keep batch_size small (e.g., 1) or per_gpu_batch
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size if args.val_batch_size>0 else 1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Model
    model = FastDepthV2()
    model.encoder = dataloaderv5_ddp.safe_load_encoder(model.encoder, args.weights_dir, args.backbone) if hasattr(dataloaderv5_ddp, "safe_load_encoder") else model.encoder
    model.decoder.apply(weights_init)
    model.to(device)

    if args.ddp:
        # remove find_unused_parameters=True unless necessary (overhead)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank) 

    criterion = DepthLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    accum_steps = args.accum_steps

    # Quick check prints
    if master_process:
        print(f"[CONFIG] epochs={args.num_epochs}, accum_steps={accum_steps}, use_amp={args.use_amp}")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(args.num_epochs):
        model.train()
        if args.ddp:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch} Train", disable=not master_process)
        step = -1
        for step, (img, depth) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            depth = depth.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred = model(img)
                loss = criterion('l1', pred, depth, epoch)

            # scale loss for accumulation
            loss_to_backward = loss / accum_steps
            scaler.scale(loss_to_backward).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            batch_count += 1
            if master_process:
                # show raw per-batch loss (smoothed by total/batch_count)
                pbar.set_postfix(loss=(total_loss / batch_count))

        # handle leftover batches (if any)
        if (step + 1) % accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Compute avg train loss PER RANK, then reduce to get global average
        local_avg_train_loss = (total_loss / batch_count) if batch_count>0 else 0.0
        if args.ddp:
            # reduce sum of averages weighted by counts: easier to send sum_loss and count
            packed = torch.tensor([total_loss, float(batch_count)], device=device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            global_sum_loss = packed[0].item()
            global_count = int(packed[1].item())
            global_avg_train_loss = global_sum_loss / global_count if global_count>0 else 0.0
        else:
            global_avg_train_loss = local_avg_train_loss

        if master_process:
            print(f"[Epoch {epoch}] train_loss={global_avg_train_loss:.5f} (local {local_avg_train_loss:.5f})")

        # -----------------------------
        # Validation: compute per-rank sums and counts, then aggregate
        # -----------------------------
        model.eval()
        local_val_sum = 0.0
        local_val_count = 0
        local_sum_d1 = 0.0  # example metric sum

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Rank {rank} Epoch {epoch} Val", disable=not master_process)
            for img, depth in vbar:
                img = img.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    pred = model(img)
                    l = criterion('l1', pred, depth, epoch)
                local_val_sum += l.item()
                # compute d1 for this batch (assume batch size 1 or small)
                cur = utils.eval_depth_for_logging(pred, depth) if hasattr(utils, "eval_depth_for_logging") else None
                if cur is not None:
                    local_sum_d1 += cur.get("d1", 0.0)
                local_val_count += 1

        # Aggregate across ranks
        if args.ddp:
            packed = torch.tensor([local_val_sum, local_sum_d1, float(local_val_count)], device=device)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            total_val_sum = packed[0].item()
            total_sum_d1 = packed[1].item()
            total_val_count = int(packed[2].item())
            global_val_loss = total_val_sum / total_val_count if total_val_count>0 else 0.0
            global_d1 = total_sum_d1 / total_val_count if total_val_count>0 else 0.0
        else:
            global_val_loss = (local_val_sum / local_val_count) if local_val_count>0 else 0.0
            global_d1 = (local_sum_d1 / local_val_count) if local_val_count>0 else 0.0

        if master_process:
            print(f"[Epoch {epoch}] val_loss={global_val_loss:.5f}, val_d1={global_d1:.5f} (total samples={total_val_count if args.ddp else local_val_count})")

    # Cleanup
    if args.ddp:
        cleanup_ddp()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="If --batch_is_global is set, this is global batch; otherwise per-GPU batch.")
    parser.add_argument("--batch_is_global", action="store_true",
                        help="Interpret --batch_size as global batch and divide by world_size to get per-GPU batch.")
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--size", type=str, default="160,128")
    parser.add_argument("--weights_dir", type=str, default="/kaggle/working/kgl_src/Weights")
    parser.add_argument("--backbone", type=str, default="mobilenetv2")
    args = parser.parse_args()

    # parse size
    if isinstance(args.size, str):
        args.size = tuple(map(int, args.size.split(",")))

    train(args)

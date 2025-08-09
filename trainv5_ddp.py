import os
import glob
import json
import torch
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

import utils
from model_v4 import FastDepthV2, weights_init
import dataloader_v5
from load_pretrained import load_pretrained_encoder
from metric_depth.util.loss import DepthLoss

torch.backends.cudnn.benchmark = True


def eval_depth(pred, target):
    valid_mask = ((target > 0) + (pred > 0)) > 0
    pred_output = pred[valid_mask]
    target_masked = target[valid_mask]
    abs_diff = (pred_output - target_masked).abs()
    RMSE = torch.sqrt(torch.mean((abs_diff).pow(2)))
    maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
    d1 = float((maxRatio < 1.25).float().mean())
    return {'d1': d1, 'rmse': RMSE.item()}


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def train_fn(rank, world_size, args):
    print(f"Start DDP process on rank {rank}.")
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Lấy đường dẫn dữ liệu
    train_paths = dataloader_v5.get_image_label_pairs(os.path.join(args.data_root, "train"))
    val_paths = dataloader_v5.get_image_label_pairs(os.path.join(args.data_root, "val"))

    # Tạo Dataset
    train_dataset = dataloader_v5.DepthDataset(train_paths, mode="train", size=args.size)
    val_dataset = dataloader_v5.DepthDataset(val_paths, mode="val", size=args.size)

    # Tạo DistributedSampler dựa trên dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Tạo DataLoader với sampler
    train_loader, val_loader = dataloader_v5.create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        num_workers=args.num_workers
    )

    # Model + pretrained encoder + init decoder
    model = FastDepthV2()
    model.encoder = load_pretrained_encoder(model.encoder, args.weights_dir, args.backbone)
    model.decoder.apply(weights_init)
    model = model.to(device)

    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = DepthLoss()

    scaler = torch.amp.GradScaler()

    best_val_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if args.load_state:
        ckpt_path = os.path.join(args.state_path, "checkpoint_best.pth")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt)

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch} Train", disable=(rank != 0))
        for input, target in pbar:
            img = input.to(device, non_blocking=True)
            depth = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast():
                pred = model(img)
                loss = criterion('l1', pred, depth, epoch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if rank == 0:
                pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        results = {'d1': 0.0, 'rmse': 0.0}
        test_loss = 0.0

        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Rank {rank} Epoch {epoch} Val", disable=(rank != 0))
            for input, target in vbar:
                img = input.to(device, non_blocking=True)
                depth = target.to(device, non_blocking=True)

                with torch.amp.autocast():
                    pred = model(img)
                    l = criterion('l1', pred, depth)

                test_loss += l.item()
                cur_results = eval_depth(pred, depth)
                for k in results:
                    results[k] += cur_results[k]

                if rank == 0:
                    vbar.set_postfix(val_loss=test_loss / (vbar.n + 1))

        # Trung bình metric trên tất cả GPU
        for key in results:
            t = torch.tensor(results[key], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            results[key] = (t / world_size).item()

        val_loss = test_loss / len(val_loader)

        if rank == 0:
            print(f"Epoch {epoch}: train_loss={avg_loss:.5f}, val_loss={val_loss:.5f}, metrics={results}")
            os.makedirs(args.state_path, exist_ok=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.state_path, f"checkpoint_best_{epoch}.pth")
                torch.save(model.module.state_dict(), ckpt_path)

                # Xóa checkpoint cũ
                for f in glob.glob(os.path.join(args.state_path, "checkpoint_best_*.pth")):
                    if f != ckpt_path:
                        os.remove(f)

            history["train_loss"].append(avg_loss)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(results)
            with open(os.path.join(args.state_path, "history.json"), "w") as f:
                json.dump(history, f, indent=2)

    cleanup_ddp()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weights_dir", type=str, default="/kaggle/working/kgl_src/Weights")
    parser.add_argument("--backbone", type=str, default="mobilenetv2")
    parser.add_argument("--data_root", type=str, default="/kaggle/input/hypdataset-v1-0/hypdataset_v1")
    parser.add_argument("--state_path", type=str, default="/kaggle/working/ours_checkpoints")
    parser.add_argument("--load_state", action="store_true")
    parser.add_argument("--size", type=tuple, default=(160, 128))
    parser.add_argument("--num_epochs", type=int, default=50)
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No GPUs found!")

    torch.multiprocessing.spawn(train_fn, args=(world_size, args), nprocs=world_size, join=True)

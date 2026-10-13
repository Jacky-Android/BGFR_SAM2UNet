"""Train the OFFICIAL SAM2UNet architecture on WHU building data.

Keeps the same data pipeline, AdamW settings, validation metrics and checkpoint
selection (macro IoU + macro boundary F1)/2 as BGFR train_boundary_whu.py.
The only structural change relative to BGFR training is removal of BGFR modules
and their boundary-supervision loss.

Run this script from the original SAM2-UNet repository directory.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from SAM2UNet import SAM2UNet
from whu_baseline_utils import BuildingDataset, structure_loss, calc_metrics


def main(args):
    if args.size <= 0 or args.size % 32 != 0:
        raise ValueError('--size must be divisible by 32, e.g. 352 or 512')
    if args.accum < 1 or args.batch_size < 1:
        raise ValueError('--accum and --batch_size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = BuildingDataset(args.train_images, args.train_masks, args.size, train=True)
    val_ds = BuildingDataset(args.val_images, args.val_masks, args.size, train=False)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == 'cuda')

    # This is exactly the original SAM2UNet model, not BGFRSAM2UNet.
    model = SAM2UNet(args.hiera_path).to(device)
    adapter_params = [p for n, p in model.named_parameters()
                      if n.startswith('encoder.') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('encoder.') and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': adapter_params, 'lr': args.encoder_lr},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_running = 0.0
        bar = tqdm(train_loader, desc=f'Train {epoch+1:03d}/{args.epochs}',
                   unit='batch', dynamic_ncols=True)
        for i, (image, gt) in enumerate(bar):
            image, gt = image.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred, aux1, aux2 = model(image)
                loss = (structure_loss(pred, gt) +
                        structure_loss(aux1, gt) +
                        structure_loss(aux2, gt))
            # Correct gradient normalization for the final partial accumulation window.
            window_start = (i // args.accum) * args.accum
            window_size = min(args.accum, len(train_loader) - window_start)
            scaler.scale(loss / window_size).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_([p for p in model.parameters()
                                          if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            loss_running += float(loss.item())
            bar.set_postfix(loss=f'{loss_running / (i+1):.4f}')
        scheduler.step()

        # Same validation metric and model selection rule used for BGFR.
        model.eval()
        sums = [0.0, 0.0, 0.0]
        total = 0
        with torch.inference_mode():
            for image, gt in tqdm(val_loader, desc='Validating',
                                  unit='batch', leave=False, dynamic_ncols=True):
                image, gt = image.to(device), gt.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits, _, _ = model(image)
                vals = calc_metrics(logits.float(), gt, tolerance=args.tolerance)
                bs = image.size(0)
                for k, v in enumerate(vals):
                    sums[k] += v * bs
                total += bs
        miou, mf1, mbf1 = (s / max(total, 1) for s in sums)
        score = 0.5 * miou + 0.5 * mbf1
        print(f'Epoch {epoch+1:03d}: train_loss={loss_running/len(train_loader):.4f}, '
              f'val_IoU={miou:.4f}, val_F1={mf1:.4f}, '
              f'val_BF1@{args.tolerance}px={mbf1:.4f}')
        if score > best_score:
            best_score = score
            ckpt_path = outdir / 'sam2unet_best.pth'
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'val_iou': miou,
                'val_f1': mf1,
                'val_boundary_f1': mbf1,
                'args': vars(args),
            }, ckpt_path)
            print(f'  New best model: {ckpt_path}')
        # Do not call every batch: only frees currently unused CUDA cache.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    print(f'Best validation score (0.5 IoU + 0.5 BF1): {best_score:.4f}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Original SAM2-UNet WHU baseline training')
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--train_images', required=True)
    ap.add_argument('--train_masks', required=True)
    ap.add_argument('--val_images', required=True)
    ap.add_argument('--val_masks', required=True)
    ap.add_argument('--output', default='./outputs/sam2_baseline')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--tolerance', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_amp', action='store_true')
    main(ap.parse_args())

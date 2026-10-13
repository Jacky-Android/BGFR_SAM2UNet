import os
# Must be set BEFORE importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from SAM2UNet_Boundary import BGFRSAM2UNet


# --------------------------------------------------------------------------- #
#  Distributed helpers
# --------------------------------------------------------------------------- #
def setup_distributed():
    """Return (device, local_rank, world_size, use_ddp, is_main)."""
    if "LOCAL_RANK" not in os.environ:
        # Plain `python train.py` -> single process, no DDP.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1, False, True

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
        return device, local_rank, world_size, True, (dist.get_rank() == 0)
    # torchrun with nproc_per_node=1 -> behave like single-GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, 0, 1, False, True


def cleanup_distributed(use_ddp):
    if use_ddp and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def allreduce_sum_(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class DistributedEvalSampler(Sampler):
    """No-padding distributed validation sampler: count each image once."""
    def __init__(self, dataset, rank, num_replicas):
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return len(range(self.rank, len(self.dataset), self.num_replicas))


class BuildingDataset(Dataset):
    """Match images/masks via identical filename stems; masks must be binary."""
    SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

    def __init__(self, images_dir, masks_dir, size=512, train=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.size = size
        self.train = train
        img_map = {p.stem: p for p in self.images_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in self.SUFFIXES}
        msk_map = {p.stem: p for p in self.masks_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in self.SUFFIXES}
        if not img_map or not msk_map:
            raise RuntimeError('Empty image or mask folder.')
        missing = sorted(set(img_map) ^ set(msk_map))
        if missing:
            raise RuntimeError(f'Image/mask stem mismatch: {missing[:10]}')
        self.samples = [(img_map[k], msk_map[k]) for k in sorted(img_map)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as im:
            image = im.convert('RGB')
        with Image.open(mask_path) as im:
            mask = im.convert('L')

        # RGB interpolation is bilinear; categorical masks MUST use nearest.
        image = TF.resize(image, [self.size, self.size],
                          interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size],
                         interpolation=InterpolationMode.NEAREST)
        if self.train:
            if random.random() < 0.5:
                image, mask = TF.hflip(image), TF.hflip(mask)
            if random.random() < 0.5:
                image, mask = TF.vflip(image), TF.vflip(mask)

        image = TF.normalize(TF.to_tensor(image),
                             mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        # PIL L-mode -> [0,1]. For 0/255 or 0/1 stored as uint8, >0 is building.
        mask = (TF.to_tensor(mask) > 0).float()
        return image, mask


# --------------------------------------------------------------------------- #
#  Losses and metrics (unchanged math)
# --------------------------------------------------------------------------- #
def structure_loss(logits, mask):
    weights = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    bce = (weights * bce).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-6)
    probs = torch.sigmoid(logits)
    inter = (probs * mask * weights).sum((2, 3))
    union = ((probs + mask) * weights).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (bce + wiou).mean()


def binary_boundary(binary_mask, kernel_size=3):
    padding = kernel_size // 2
    dilated = F.max_pool2d(binary_mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-binary_mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0, 1)


def edge_loss(edge_logits_half, mask_full, positive_weight=5.0):
    half_mask = F.interpolate(mask_full, size=edge_logits_half.shape[-2:],
                              mode='nearest')
    edge_gt = binary_boundary(half_mask, kernel_size=3)
    pos = edge_logits_half.new_tensor(positive_weight)
    bce = F.binary_cross_entropy_with_logits(edge_logits_half, edge_gt,
                                             pos_weight=pos)
    prob = torch.sigmoid(edge_logits_half)
    inter = (prob * edge_gt).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (
        prob.sum((2, 3)) + edge_gt.sum((2, 3)) + 1
    )
    return 0.5 * bce + 0.5 * dice.mean()


def calc_metrics(logits, gt, tolerance=2):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    axes = (2, 3)
    intersection = (pred * gt).sum(axes)
    union = (pred + gt - pred * gt).sum(axes)
    iou = torch.where(union > 0, intersection / union.clamp_min(1),
                      torch.ones_like(union))
    denom = pred.sum(axes) + gt.sum(axes)
    f1 = torch.where(denom > 0, 2 * intersection / denom.clamp_min(1),
                     torch.ones_like(denom))
    pe = binary_boundary(pred)
    ge = binary_boundary(gt)
    k = 2 * tolerance + 1
    ge_near = F.max_pool2d(ge, k, stride=1, padding=tolerance)
    pe_near = F.max_pool2d(pe, k, stride=1, padding=tolerance)
    count_p, count_g = pe.sum(axes), ge.sum(axes)
    precision = ((pe * ge_near).sum(axes) / count_p.clamp_min(1))
    recall = ((ge * pe_near).sum(axes) / count_g.clamp_min(1))
    bf1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    bf1 = torch.where((count_p == 0) & (count_g == 0),
                       torch.ones_like(bf1), bf1)
    return iou.mean().item(), f1.mean().item(), bf1.mean().item()


class BinaryConfusion:
    """Aggregate dataset-level binary confusion matrix."""
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    @torch.no_grad()
    def update(self, logits, gt):
        pred = torch.sigmoid(logits.float()) >= 0.5
        truth = gt > 0.5
        self.tp += int((pred & truth).sum().item())
        self.fp += int((pred & ~truth).sum().item())
        self.tn += int((~pred & ~truth).sum().item())
        self.fn += int((~pred & truth).sum().item())

    def metrics(self):
        tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        pixels = tp + fp + tn + fn
        acc = (tp + tn) / max(pixels, 1)
        iou_building = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        iou_background = tn / (tn + fp + fn) if tn + fp + fn else 1.0
        miou = (iou_building + iou_background) / 2.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        return dict(acc=acc, building_iou=iou_building,
                    background_iou=iou_background, miou=miou, f1=f1)

    def sync_dist(self, device):
        """All-reduce tp/fp/tn/fn across ranks (for global validation metrics)."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.tp, self.fp, self.tn, self.fn],
                         dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        self.tp, self.fp, self.tn, self.fn = [int(x) for x in t.tolist()]


# --------------------------------------------------------------------------- #
#  Main training routine
# --------------------------------------------------------------------------- #
def train(args):
    if args.size <= 0 or args.size % 32:
        raise ValueError('--size must be a positive multiple of 32 (e.g. 352, 512).')
    if args.accum < 1 or args.batch_size < 1:
        raise ValueError('--accum and --batch_size must be positive.')

    # ---- Distributed setup ----
    device, local_rank, world_size, use_ddp, is_main = setup_distributed()

    def log(*a, **kw):
        if is_main:
            print(*a, **kw, flush=True)

    # Seed each rank differently; add rank offset so DDP ranks do not duplicate crops.
    seed = args.seed + (dist.get_rank() if use_ddp else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    if use_ddp:
        log(f'DDP active: world_size={world_size}, this rank local_rank={local_rank}')
        log(f'Per-GPU batch={args.batch_size}; effective global batch='
            f'{args.batch_size * world_size * args.accum}')
    elif device.type == 'cuda':
        log(f'Single GPU: {torch.cuda.get_device_name(0)}')
    else:
        log('WARNING: CUDA unavailable; training on CPU.')

    if args.multi_gpu and not use_ddp:
        log('WARNING: --multi_gpu is deprecated and ignored. '
            'Use `torchrun --nproc_per_node=2 ...` for multi-GPU DDP training.')

    # ---- Datasets ----
    train_ds = BuildingDataset(args.train_images, args.train_masks, args.size, train=True)
    val_ds = BuildingDataset(args.val_images, args.val_masks, args.size, train=False)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                       rank=local_rank, shuffle=True) if use_ddp else None
    # DistributedSampler pads validation to a multiple of world_size, which
    # duplicates images whenever len(val_ds) is not evenly divisible.
    val_sampler = DistributedEvalSampler(val_ds, rank=local_rank,
                                         num_replicas=world_size) if use_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=use_ddp,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
    )

    log(f'Samples: train={len(train_ds)}, val={len(val_ds)}; '
        f'batches per epoch per rank: train={len(train_loader)}, val={len(val_loader)}')

    # ---- Model ----
    model = BGFRSAM2UNet(
        args.hiera_path, enable_edge_gate=not args.disable_edge_gate
    ).to(device)

    # Build optimizer from the *unwrapped* model; parameter objects are shared
    # with the DDP wrapper afterwards, so names remain `encoder.*`.
    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith('encoder.') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('encoder.') and p.requires_grad]
    optim = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    amp = device.type == 'cuda' and not args.no_amp
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # ---- Output ----
    output = Path(args.output)
    if is_main:
        output.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()  # ensure dir exists before any rank tries to write
    history_path = output / 'history.csv'
    columns = ['epoch', 'train_loss', 'train_acc', 'train_building_iou',
               'train_miou', 'val_loss', 'val_acc', 'val_building_iou',
               'val_background_iou', 'val_miou', 'val_f1',
               'val_macro_iou', 'val_macro_f1', 'val_boundary_f1',
               'lr', 'epoch_seconds']

    best_score = -1.0
    log(f'Progress log will be saved to {history_path}')

    csvfile = history_path.open('w', newline='', encoding='utf-8') if is_main else None
    writer = csv.DictWriter(csvfile, fieldnames=columns) if is_main else None
    if is_main:
        writer.writeheader()
        csvfile.flush()

    try:
        for epoch in range(args.epochs):
            started = time.perf_counter()
            epoch_lr = optim.param_groups[1]['lr']  # LR actually used this epoch

            # DDP: reshuffle every epoch
            if use_ddp:
                train_sampler.set_epoch(epoch)

            # ---------------- Training ----------------
            model.train()
            optim.zero_grad(set_to_none=True)
            train_meter = BinaryConfusion()
            train_loss_total = 0.0
            train_samples = 0

            train_bar = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{args.epochs} [Train]',
                leave=False, dynamic_ncols=True,
                disable=(args.no_progress_bar or not is_main),
                mininterval=0.5,
            )
            for i, (image, gt) in enumerate(train_bar):
                image = image.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    pred, aux1, aux2, edge = model(image)
                    seg = (structure_loss(pred, gt) +
                           structure_loss(aux1, gt) +
                           structure_loss(aux2, gt))
                    boundary = edge_loss(edge, gt, args.pos_weight)
                    loss = seg + args.edge_weight * boundary

                # Correct last accumulation window if the number of batches
                # is not divisible by args.accum (e.g. final window of 1).
                window_start = (i // args.accum) * args.accum
                window_size = min(args.accum, len(train_loader) - window_start)
                scaler.scale(loss / window_size).backward()
                if (i + 1) % args.accum == 0 or i + 1 == len(train_loader):
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)

                bs = image.size(0)
                train_samples += bs
                train_loss_total += float(loss.detach().item()) * bs
                train_meter.update(pred.detach(), gt)

                if (i + 1) % args.progress_every == 0 or i + 1 == len(train_loader):
                    stats = train_meter.metrics()
                    avg_loss = train_loss_total / max(train_samples, 1)
                    if is_main:
                        train_bar.set_postfix(
                            loss=f'{avg_loss:.3f}',
                            acc=f'{stats["acc"]*100:.2f}%',
                            buildingIoU=f'{stats["building_iou"]*100:.2f}%')
                        if args.no_progress_bar:
                            print(f'Epoch {epoch+1}/{args.epochs} Train '
                                  f'{i+1}/{len(train_loader)}: loss={avg_loss:.4f}, '
                                  f'acc={stats["acc"]*100:.2f}%, '
                                  f'buildingIoU={stats["building_iou"]*100:.2f}%',
                                  flush=True)
            if is_main:
                train_bar.close()
            # Aggregate train statistics rather than reporting only GPU 0.
            if use_ddp:
                train_meter.sync_dist(device)
                t_train = torch.tensor([train_loss_total, train_samples],
                                       dtype=torch.float64, device=device)
                allreduce_sum_(t_train)
                train_loss_total, train_samples = t_train.tolist()
            scheduler.step()

            # ---------------- Validation ----------------
            model.eval()
            val_meter = BinaryConfusion()
            val_loss_total = 0.0
            val_samples = 0
            bf1_total = 0.0
            macro_iou_total = 0.0
            macro_f1_total = 0.0
            # Validation uses the unwrapped module, since the no-padding
            # sampler can give different numbers of batches per rank.
            # DDP forward can perform synchronization collectives; plain
            # module forward under no_grad does not.
            eval_model = model.module if use_ddp else model

            val_bar = tqdm(
                val_loader,
                desc=f'Epoch {epoch+1}/{args.epochs} [Val]',
                leave=False, dynamic_ncols=True,
                disable=(args.no_progress_bar or not is_main),
                mininterval=0.5,
            )
            with torch.no_grad():
                for i, (image, gt) in enumerate(val_bar):
                    image = image.to(device, non_blocking=True)
                    gt = gt.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, enabled=amp):
                        logits, aux1, aux2, edge = eval_model(image)
                        val_seg = (structure_loss(logits, gt) +
                                   structure_loss(aux1, gt) +
                                   structure_loss(aux2, gt))
                        val_boundary = edge_loss(edge, gt, args.pos_weight)
                        val_loss = val_seg + args.edge_weight * val_boundary

                    bs = image.size(0)
                    val_samples += bs
                    val_loss_total += float(val_loss.item()) * bs
                    val_meter.update(logits, gt)
                    batch_iou, batch_f1, batch_bf1 = calc_metrics(
                        logits.float(), gt, tolerance=args.tolerance)
                    macro_iou_total += batch_iou * bs
                    macro_f1_total += batch_f1 * bs
                    bf1_total += batch_bf1 * bs

                    if (i + 1) % args.progress_every == 0 or i + 1 == len(val_loader):
                        stats = val_meter.metrics()
                        avg_val_loss = val_loss_total / max(val_samples, 1)
                        if is_main:
                            val_bar.set_postfix(
                                loss=f'{avg_val_loss:.3f}',
                                acc=f'{stats["acc"]*100:.2f}%',
                                mIoU=f'{stats["miou"]*100:.2f}%')
                            if args.no_progress_bar:
                                print(f'Epoch {epoch+1}/{args.epochs} Val '
                                      f'{i+1}/{len(val_loader)}: loss={avg_val_loss:.4f}, '
                                      f'acc={stats["acc"]*100:.2f}%, '
                                      f'mIoU={stats["miou"]*100:.2f}%', flush=True)
            if is_main:
                val_bar.close()

            # ---- Cross-rank aggregation of validation metrics ----
            if use_ddp:
                # Loss/sample/macro metric sums, each image counted once.
                t = torch.tensor([val_loss_total, val_samples, bf1_total,
                                  macro_iou_total, macro_f1_total],
                                 dtype=torch.float64, device=device)
                allreduce_sum_(t)
                val_loss_total, val_samples, bf1_total, macro_iou_total, macro_f1_total = t.tolist()
                # Confusion matrix
                val_meter.sync_dist(device)

            tr = train_meter.metrics()
            va = val_meter.metrics()
            train_loss = train_loss_total / max(train_samples, 1)
            val_loss = val_loss_total / max(val_samples, 1)
            mbf1 = bf1_total / max(val_samples, 1)
            macro_iou = macro_iou_total / max(val_samples, 1)
            macro_f1 = macro_f1_total / max(val_samples, 1)
            # Matches the original single-GPU BGFR + SAM2-UNet baseline
            # checkpoint criterion: macro building IoU + macro BF1.
            score = 0.5 * macro_iou + 0.5 * mbf1
            duration = time.perf_counter() - started

            row = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': tr['acc'],
                'train_building_iou': tr['building_iou'],
                'train_miou': tr['miou'],
                'val_loss': val_loss,
                'val_acc': va['acc'],
                'val_building_iou': va['building_iou'],
                'val_background_iou': va['background_iou'],
                'val_miou': va['miou'],
                'val_f1': va['f1'],
                'val_macro_iou': macro_iou,
                'val_macro_f1': macro_f1,
                'val_boundary_f1': mbf1,
                'lr': epoch_lr,
                'epoch_seconds': duration,
            }

            if is_main:
                writer.writerow(row)
                csvfile.flush()
                log(f'\nEpoch [{epoch+1:03d}/{args.epochs:03d}] {duration:.1f}s | '
                    f'train loss={train_loss:.4f} acc={tr["acc"]*100:.2f}% | '
                    f'val loss={val_loss:.4f} acc={va["acc"]*100:.2f}% '
                    f'mIoU={va["miou"]*100:.2f}% '
                    f'buildingIoU(micro)={va["building_iou"]*100:.2f}% '
                    f'F1(micro)={va["f1"]*100:.2f}% '
                    f'IoU(macro)={macro_iou*100:.2f}% '
                    f'BF1@{args.tolerance}px(macro)={mbf1*100:.2f}%')

                if score > best_score:
                    best_score = score
                    state = (model.module.state_dict()
                             if isinstance(model, DDP) else model.state_dict())
                    torch.save({
                        'model': state,
                        'epoch': epoch + 1,
                        'val_iou': macro_iou,  # compatible with original BGFR checkpoint naming
                        'val_building_iou_micro': va['building_iou'],
                        'val_miou': va['miou'],
                        'val_acc': va['acc'],
                        'val_f1': va['f1'],
                        'val_macro_iou': macro_iou,
                        'val_macro_f1': macro_f1,
                        'val_boundary_f1': mbf1,
                        'args': vars(args),
                    }, output / 'bgfr_best.pth')
                    log(f'  New best checkpoint saved: {output / "bgfr_best.pth"}')

            if device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        if is_main and csvfile is not None:
            csvfile.close()
        if is_main:
            print(f'Finished. Best score (0.5 macro IoU + 0.5 macro BF1): {best_score:.4f}')
            print(f'Epoch history: {history_path}')
        cleanup_distributed(use_ddp)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hiera_path', required=True)
    ap.add_argument('--train_images', required=True)
    ap.add_argument('--train_masks', required=True)
    ap.add_argument('--val_images', required=True)
    ap.add_argument('--val_masks', required=True)
    ap.add_argument('--output', default='./bgfr_checkpoints')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=45)
    ap.add_argument('--batch_size', type=int, default=1,
                    help='Per-GPU batch size (global = batch_size * world_size).')
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--workers', type=int, default=2,
                    help='DataLoader workers per rank. Keep small on Kaggle to '
                         'avoid system-RAM OOM.')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--edge_weight', type=float, default=0.2)
    ap.add_argument('--pos_weight', type=float, default=5.0)
    ap.add_argument('--tolerance', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--multi_gpu', action='store_true',
                    help='DEPRECATED. Use torchrun --nproc_per_node=N for multi-GPU.')
    ap.add_argument('--disable_edge_gate', action='store_true',
                    help='Ablation: remove edge-guided gating.')
    ap.add_argument('--progress_every', type=int, default=10,
                    help='Refresh metrics every N batches (default: 10).')
    ap.add_argument('--no_progress_bar', action='store_true',
                    help='Print periodic log lines instead of tqdm bars.')
    args = ap.parse_args()
    if args.progress_every < 1:
        ap.error('--progress_every must be >= 1')
    train(args)
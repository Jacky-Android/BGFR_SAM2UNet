"""Train BGFRSAM2UNet on a binary building dataset (WHU-like folder layout).

Dependencies: original SAM2-UNet repository, PyTorch, torchvision, Pillow, numpy.
Example invocation is documented in README_BGFR.md.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR

from SAM2UNet_Boundary import BGFRSAM2UNet


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


def structure_loss(logits, mask):
    """Official weighted BCE + weighted IoU, with modern reduction syntax."""
    weights = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction='none')
    bce = (weights * bce).sum((2, 3)) / weights.sum((2, 3)).clamp_min(1e-6)
    probs = torch.sigmoid(logits)
    inter = (probs * mask * weights).sum((2, 3))
    union = ((probs + mask) * weights).sum((2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (bce + wiou).mean()


def binary_boundary(binary_mask, kernel_size=3):
    """Inside+outside morphological boundary, output values {0,1}."""
    padding = kernel_size // 2
    dilated = F.max_pool2d(binary_mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-binary_mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0, 1)


def edge_loss(edge_logits_half, mask_full, positive_weight=5.0):
    # Derive boundary at the SAME scale as edge logits; no bilinear label resize.
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
    """Foreground IoU/F1 and boundary F1 at `tolerance` pixels (per image)."""
    pred = (torch.sigmoid(logits) >= 0.5).float()
    axes = (2, 3)
    intersection = (pred * gt).sum(axes)
    union = (pred + gt - pred * gt).sum(axes)
    # Both-empty mask: perfect agreement, by explicit convention.
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


def train(args):
    if args.size % 32:
        raise ValueError('--size must be divisible by 32 (e.g. 352, 512).')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = BuildingDataset(args.train_images, args.train_masks, args.size, train=True)
    val_ds = BuildingDataset(args.val_images, args.val_masks, args.size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == 'cuda')

    model = BGFRSAM2UNet(
        args.hiera_path, enable_edge_gate=not args.disable_edge_gate
    ).to(device)
    # Freeze policy is inherited from the official SAM2-UNet. Adapters are trainable.
    encoder_params = [p for n, p in model.named_parameters()
                      if n.startswith('encoder.') and p.requires_grad]
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith('encoder.') and p.requires_grad]
    optim = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.encoder_lr},
        {'params': other_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-7)
    amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss_running = 0.0
        for i, (image, gt) in enumerate(train_loader):
            image, gt = image.to(device), gt.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred, aux1, aux2, edge = model(image)
                seg = (structure_loss(pred, gt) +
                       structure_loss(aux1, gt) +
                       structure_loss(aux2, gt))
                boundary = edge_loss(edge, gt, args.pos_weight)
                loss = seg + args.edge_weight * boundary
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            loss_running += loss.item()

        scheduler.step()
        model.eval()
        sums = [0.0, 0.0, 0.0]
        total = 0
        with torch.no_grad():
            for image, gt in val_loader:
                image, gt = image.to(device), gt.to(device)
                with torch.autocast(device_type=device.type, enabled=amp):
                    logits, _, _, _ = model(image)
                metrics = calc_metrics(logits.float(), gt, tolerance=args.tolerance)
                batch_size = image.size(0)
                for k in range(3):
                    sums[k] += metrics[k] * batch_size
                total += batch_size
        miou, mf1, mbf1 = (s / max(total, 1) for s in sums)
        score = 0.5 * miou + 0.5 * mbf1
        print(f'Epoch {epoch+1:03d}: loss={loss_running/len(train_loader):.4f}, '
              f'IoU={miou:.4f}, F1={mf1:.4f}, BF1@{args.tolerance}px={mbf1:.4f}')
        if score > best_score:
            best_score = score
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch + 1,
                'val_iou': miou,
                'val_f1': mf1,
                'val_boundary_f1': mbf1,
                'args': vars(args),
            }, Path(args.output) / 'bgfr_best.pth')
    print('Best selection score (0.5 IoU + 0.5 BF1):', best_score)


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
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr', type=float, default=5e-5)
    ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--edge_weight', type=float, default=0.2)
    ap.add_argument('--pos_weight', type=float, default=5.0)
    ap.add_argument('--tolerance', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--disable_edge_gate', action='store_true',
                    help='Ablation: remove edge-guided gating, keep same detail/refine path')
    train(ap.parse_args())

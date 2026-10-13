"""Shared WHU preprocessing and metrics: identical to BGFR training pipeline.

This is independent of BGFR model code.
"""
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


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


